"""Export templates for various languages and frameworks."""

from .base import BaseTemplate
from .python_playwright import PythonPlaywrightTemplate
from .python_selenium import PythonSeleniumTemplate
from .typescript_playwright import TypeScriptPlaywrightTemplate
from .java_selenium import JavaSeleniumTemplate
from .csharp_selenium import CSharpSeleniumTemplate
from .ruby_capybara import RubyCapybaraTemplate
from .go_rod import GoRodTemplate

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
