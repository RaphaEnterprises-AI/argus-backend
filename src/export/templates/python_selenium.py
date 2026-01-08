"""Python Selenium export template."""

from .base import BaseTemplate, TestSpec, TestStep, TestAssertion


class PythonSeleniumTemplate(BaseTemplate):
    """Template for Python Selenium tests."""

    language = "python"
    framework = "selenium"
    file_extension = ".py"
    indent = "    "

    def generate_imports(self, test_spec: TestSpec) -> str:
        """Generate Python imports."""
        imports = [
            "import pytest",
            "from selenium import webdriver",
            "from selenium.webdriver.common.by import By",
            "from selenium.webdriver.common.keys import Keys",
            "from selenium.webdriver.support.ui import WebDriverWait",
            "from selenium.webdriver.support import expected_conditions as EC",
            "from selenium.webdriver.support.ui import Select",
            "from selenium.webdriver.common.action_chains import ActionChains",
        ]
        return "\n".join(imports)

    def generate_class_header(self, test_spec: TestSpec) -> str:
        """Generate pytest class header."""
        class_name = self.config.get("test_class_name") or f"Test{self.to_pascal_case(test_spec.name)}"
        method_name = f"test_{self.to_snake_case(test_spec.name)}"

        lines = [
            "",
            f'class {class_name}:',
            f'    """Generated from Argus test spec: {test_spec.id}"""',
            "",
            "    @pytest.fixture(autouse=True)",
            "    def setup(self, driver, base_url):",
            "        self.driver = driver",
            "        self.base_url = base_url",
            "        self.wait = WebDriverWait(driver, 10)",
            "",
            f"    def {method_name}(self):",
        ]

        if test_spec.description:
            lines.append(f'        """{test_spec.description}"""')

        return "\n".join(lines)

    def _get_locator(self, target: str) -> str:
        """Convert selector to Selenium locator."""
        if not target:
            return 'By.CSS_SELECTOR, ""'

        if target.startswith("#"):
            return f'By.ID, "{target[1:]}"'
        elif target.startswith("."):
            return f'By.CLASS_NAME, "{target[1:]}"'
        elif target.startswith("//"):
            return f'By.XPATH, "{self.escape_string(target)}"'
        elif target.startswith("name="):
            return f'By.NAME, "{target[5:]}"'
        else:
            return f'By.CSS_SELECTOR, "{self.escape_string(target)}"'

    def generate_step_code(self, step: TestStep, index: int) -> str:
        """Generate code for a single step."""
        action = step.action.lower()
        target = step.target or ""
        value = step.value or ""
        locator = self._get_locator(target)

        comment = f"        # Step {index + 1}: {step.description or f'{action} {target}'.strip()}"
        code_line = ""

        if action == "goto":
            url = target if target.startswith("http") else f'f"{{self.base_url}}{target}"'
            if target.startswith("http"):
                code_line = f'        self.driver.get("{self.escape_string(target)}")'
            else:
                code_line = f'        self.driver.get(f"{{self.base_url}}{target}")'

        elif action == "click":
            code_line = f"""        element = self.wait.until(
            EC.element_to_be_clickable(({locator}))
        )
        element.click()"""

        elif action == "fill":
            code_line = f"""        element = self.wait.until(
            EC.presence_of_element_located(({locator}))
        )
        element.clear()
        element.send_keys("{self.escape_string(value)}")"""

        elif action == "type":
            code_line = f"""        element = self.wait.until(
            EC.presence_of_element_located(({locator}))
        )
        element.send_keys("{self.escape_string(value)}")"""

        elif action == "select":
            code_line = f"""        element = self.wait.until(
            EC.presence_of_element_located(({locator}))
        )
        Select(element).select_by_visible_text("{self.escape_string(value)}")"""

        elif action == "hover":
            code_line = f"""        element = self.wait.until(
            EC.presence_of_element_located(({locator}))
        )
        ActionChains(self.driver).move_to_element(element).perform()"""

        elif action == "wait":
            timeout = step.timeout or 10000
            code_line = f"""        self.wait.until(
            EC.presence_of_element_located(({locator}))
        )"""

        elif action == "scroll":
            if value:
                parts = value.split(",")
                x = parts[0] if len(parts) > 0 else "0"
                y = parts[1] if len(parts) > 1 else "0"
                code_line = f'        self.driver.execute_script("window.scrollBy({x}, {y})")'
            else:
                code_line = '        self.driver.execute_script("window.scrollBy(0, 300)")'

        elif action == "press_key":
            key = value or target or "ENTER"
            code_line = f"""        element = self.driver.switch_to.active_element
        element.send_keys(Keys.{key.upper()})"""

        elif action == "screenshot":
            code_line = f'        self.driver.save_screenshot("screenshot_{index}.png")'

        elif action == "double_click":
            code_line = f"""        element = self.wait.until(
            EC.element_to_be_clickable(({locator}))
        )
        ActionChains(self.driver).double_click(element).perform()"""

        else:
            code_line = f'        # Unknown action: {action}'

        return f"{comment}\n{code_line}\n"

    def generate_assertion_code(self, assertion: TestAssertion) -> str:
        """Generate code for a single assertion."""
        assertion_type = assertion.type.lower()
        target = assertion.target or ""
        expected = assertion.expected or ""
        locator = self._get_locator(target)

        if assertion_type == "element_visible":
            return f"""        element = self.wait.until(
            EC.visibility_of_element_located(({locator}))
        )
        assert element.is_displayed()"""

        elif assertion_type == "element_hidden":
            return f"""        try:
            element = self.driver.find_element({locator})
            assert not element.is_displayed()
        except:
            pass  # Element not found = hidden"""

        elif assertion_type == "text_contains":
            return f"""        element = self.wait.until(
            EC.presence_of_element_located(({locator}))
        )
        assert "{self.escape_string(expected)}" in element.text"""

        elif assertion_type == "text_equals":
            return f"""        element = self.wait.until(
            EC.presence_of_element_located(({locator}))
        )
        assert element.text == "{self.escape_string(expected)}\""""

        elif assertion_type in ("url_contains", "url_matches"):
            return f'        assert "{self.escape_string(expected)}" in self.driver.current_url'

        elif assertion_type == "value_equals":
            return f"""        element = self.wait.until(
            EC.presence_of_element_located(({locator}))
        )
        assert element.get_attribute("value") == "{self.escape_string(expected)}\""""

        elif assertion_type == "title_contains":
            return f'        assert "{self.escape_string(expected)}" in self.driver.title'

        else:
            return f'        # Unknown assertion type: {assertion_type}'

    def generate_class_footer(self) -> str:
        """Generate class footer."""
        return ""

    def _generate_assertions_header(self) -> str:
        """Generate assertions section header."""
        return "        # Assertions"
