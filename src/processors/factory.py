"""Factory for creating language processors."""
from typing import Dict, Type, Optional
from pathlib import Path

from .base import LanguageProcessor
from .python.processor import PythonProcessor
from .javascript.processor import JavaScriptProcessor
from .typescript.processor import TypeScriptProcessor
from .java.processor import JavaProcessor
from .kotlin.processor import KotlinProcessor
from .html.processor import HTMLProcessor
from .scss.processor import SCSSProcessor


class LanguageProcessorFactory:
    """Factory for creating language-specific processors."""
    
    _processors: Dict[str, Type[LanguageProcessor]] = {}
    
    @classmethod
    def register_processor(cls, language: str, processor_class: Type[LanguageProcessor]):
        """Register a processor class for a language.
        
        Args:
            language: Language name.
            processor_class: Processor class implementing LanguageProcessor.
        """
        cls._processors[language] = processor_class
    
    @classmethod
    def create_processor(cls, language: str) -> LanguageProcessor:
        """Create a processor instance for a language.
        
        Args:
            language: Language name.
            
        Returns:
            Processor instance.
            
        Raises:
            ValueError: If language is not supported.
        """
        if language not in cls._processors:
            raise ValueError(f"Unsupported language: {language}")
        return cls._processors[language]()
    
    @classmethod
    def get_processor_for_file(cls, file_path: Path) -> Optional[LanguageProcessor]:
        """Get processor for a file based on its extension.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Processor instance or None if not supported.
        """
        ext = file_path.suffix.lower()
        
        # Map extensions to languages
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.kts': 'kotlin',
            '.html': 'html',
            '.htm': 'html',
            '.scss': 'scss',
            '.sass': 'scss',
            '.css': 'scss',  # Treat CSS as SCSS
        }
        
        language = ext_to_lang.get(ext)
        if language and language in cls._processors:
            return cls.create_processor(language)
        
        return None
    
    @classmethod
    def get_supported_languages(cls) -> list:
        """Get list of supported languages.
        
        Returns:
            List of supported language names.
        """
        return list(cls._processors.keys())


# Register all processors
LanguageProcessorFactory.register_processor("python", PythonProcessor)
LanguageProcessorFactory.register_processor("javascript", JavaScriptProcessor)
LanguageProcessorFactory.register_processor("typescript", TypeScriptProcessor)
LanguageProcessorFactory.register_processor("java", JavaProcessor)
LanguageProcessorFactory.register_processor("kotlin", KotlinProcessor)
LanguageProcessorFactory.register_processor("html", HTMLProcessor)
LanguageProcessorFactory.register_processor("scss", SCSSProcessor)
