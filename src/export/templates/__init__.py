"""Export templates for various languages and frameworks."""

from .base import BaseTemplate
from .csharp_selenium import CSharpSeleniumTemplate
from .go_rod import GoRodTemplate
from .java_selenium import JavaSeleniumTemplate
from .python_playwright import PythonPlaywrightTemplate
from .python_selenium import PythonSeleniumTemplate
from .ruby_capybara import RubyCapybaraTemplate
from .typescript_playwright import TypeScriptPlaywrightTemplate

__all__ = [
    "BaseTemplate",
    "PythonPlaywrightTemplate",
    "PythonSeleniumTemplate",
    "TypeScriptPlaywrightTemplate",
    "JavaSeleniumTemplate",
    "CSharpSeleniumTemplate",
    "RubyCapybaraTemplate",
    "GoRodTemplate",
]
