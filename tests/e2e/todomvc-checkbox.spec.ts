import { test, expect } from '@playwright/test';

test.describe('TodoMVC Checkbox Toggle', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the TodoMVC application
    await page.goto('https://demo.playwright.dev/todomvc/#/');

    // Wait for the app to be fully loaded
    await page.waitForLoadState('networkidle');
  });

  test('should successfully toggle todo checkbox without null reference error', async ({ page }) => {
    // Add a new todo item
    const todoInput = page.locator('.new-todo');
    await todoInput.fill('Test todo item');
    await todoInput.press('Enter');

    // Wait for the todo item to appear
    await page.waitForSelector('.todo-list li');

    // Get the checkbox for the newly created todo
    const todoCheckbox = page.locator('.todo-list li:first-child input[type="checkbox"]');

    // Verify the checkbox exists and is not checked initially
    await expect(todoCheckbox).toBeVisible();
    await expect(todoCheckbox).not.toBeChecked();

    // Toggle the checkbox - this should not throw "Cannot read property checked of null"
    await todoCheckbox.click();

    // Verify the checkbox is now checked
    await expect(todoCheckbox).toBeChecked();

    // Verify the todo item has the completed class
    const todoItem = page.locator('.todo-list li:first-child');
    await expect(todoItem).toHaveClass(/completed/);

    // Toggle it back to unchecked
    await todoCheckbox.click();

    // Verify the checkbox is unchecked again
    await expect(todoCheckbox).not.toBeChecked();

    // Verify the completed class is removed
    await expect(todoItem).not.toHaveClass(/completed/);
  });

  test('should handle multiple todo checkbox toggles without errors', async ({ page }) => {
    // Add multiple todo items
    const todoInput = page.locator('.new-todo');
    const todoTexts = ['First todo', 'Second todo', 'Third todo'];

    for (const text of todoTexts) {
      await todoInput.fill(text);
      await todoInput.press('Enter');
      // Small delay to ensure item is added
      await page.waitForTimeout(100);
    }

    // Wait for all todos to be visible
    await page.waitForSelector('.todo-list li');
    const todoItems = page.locator('.todo-list li');
    await expect(todoItems).toHaveCount(3);

    // Toggle each checkbox multiple times
    for (let i = 0; i < 3; i++) {
      const checkbox = page.locator(`.todo-list li:nth-child(${i + 1}) input[type="checkbox"]`);

      // First toggle - check it
      await checkbox.click();
      await expect(checkbox).toBeChecked();

      // Second toggle - uncheck it
      await checkbox.click();
      await expect(checkbox).not.toBeChecked();
    }
  });

  test('should handle rapid checkbox clicks without null reference errors', async ({ page }) => {
    // Add a todo item
    const todoInput = page.locator('.new-todo');
    await todoInput.fill('Rapid click test');
    await todoInput.press('Enter');

    // Wait for the todo to appear
    await page.waitForSelector('.todo-list li');

    const checkbox = page.locator('.todo-list li:first-child input[type="checkbox"]');

    // Perform rapid clicks
    for (let i = 0; i < 5; i++) {
      await checkbox.click();
      // Small delay between clicks to allow DOM updates
      await page.waitForTimeout(50);
    }

    // After odd number of clicks, checkbox should be checked
    await expect(checkbox).toBeChecked();
  });

  test('should verify checkbox element exists before toggling', async ({ page }) => {
    // Add a todo
    const todoInput = page.locator('.new-todo');
    await todoInput.fill('Existence check test');
    await todoInput.press('Enter');

    // Wait for todo list to be visible
    await page.waitForSelector('.todo-list');

    // Verify the checkbox exists in the DOM
    const checkbox = page.locator('.todo-list li:first-child input[type="checkbox"]');
    const isVisible = await checkbox.isVisible();
    expect(isVisible).toBe(true);

    // Verify we can get the checked state without errors
    const isChecked = await checkbox.isChecked();
    expect(typeof isChecked).toBe('boolean');

    // Toggle and verify state changes
    await checkbox.click();
    const newCheckedState = await checkbox.isChecked();
    expect(newCheckedState).toBe(!isChecked);
  });
});
