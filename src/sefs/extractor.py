"""
SEFS Content Extractor
Extracts text content from PDF and text-based files for semantic analysis.
"""

import os
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("sefs.extractor")


class ContentExtractor:
    """Extracts text content from various file types."""

    ENCODING_FALLBACKS = ["utf-8", "latin-1", "cp1252", "ascii"]

    def __init__(self):
        self._pdf_available = False
        try:
            import PyPDF2
            self._pdf_available = True
        except ImportError:
            logger.warning("PyPDF2 not installed. PDF extraction disabled.")

    def extract(self, file_path: str) -> dict:
        """
        Extract content and metadata from a file.

        Returns:
            dict with keys: path, name, extension, content, size, modified, error
        """
        file_path = Path(file_path)
        metadata = {
            "path": str(file_path),
            "name": file_path.name,
            "extension": file_path.suffix.lower(),
            "content": "",
            "size": 0,
            "modified": "",
            "created": "",
            "error": None,
        }

        try:
            stat = file_path.stat()
            metadata["size"] = stat.st_size
            metadata["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            metadata["created"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        except OSError as e:
            metadata["error"] = f"Cannot stat file: {e}"
            return metadata

        ext = file_path.suffix.lower()

        if ext == ".pdf":
            metadata["content"] = self._extract_pdf(file_path, metadata)
        elif ext in (".txt", ".md", ".rst", ".csv", ".log", ".json"):
            metadata["content"] = self._extract_text(file_path, metadata)
        else:
            metadata["error"] = f"Unsupported file extension: {ext}"

        return metadata

    def _extract_pdf(self, file_path: Path, metadata: dict) -> str:
        """Extract text from a PDF file."""
        if not self._pdf_available:
            metadata["error"] = "PyPDF2 not available"
            return ""

        try:
            import PyPDF2

            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                metadata["page_count"] = len(reader.pages)

                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")

            content = "\n".join(text_parts).strip()

            if not content:
                metadata["error"] = "PDF contains no extractable text (possibly scanned/image-based)"

            return content

        except Exception as e:
            metadata["error"] = f"PDF extraction error: {e}"
            logger.error(f"Failed to extract PDF {file_path}: {e}")
            return ""

    def _extract_text(self, file_path: Path, metadata: dict) -> str:
        """Extract text from a plain text file with encoding fallback."""
        for encoding in self.ENCODING_FALLBACKS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                metadata["encoding"] = encoding
                return content.strip()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                metadata["error"] = f"Text extraction error: {e}"
                logger.error(f"Failed to extract text from {file_path}: {e}")
                return ""

        metadata["error"] = "Could not decode file with any supported encoding"
        return ""

    def is_supported(self, file_path: str) -> bool:
        """Check if a file type is supported for extraction."""
        ext = Path(file_path).suffix.lower()
        return ext in (".pdf", ".txt", ".md", ".rst", ".csv", ".log", ".json")

    def batch_extract(self, file_paths: list) -> list:
        """Extract content from multiple files."""
        results = []
        for fp in file_paths:
            try:
                result = self.extract(fp)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch extraction error for {fp}: {e}")
                results.append({
                    "path": str(fp),
                    "name": Path(fp).name,
                    "content": "",
                    "error": str(e),
                })
        return results
