const bcrypt = require('bcryptjs');

// Pre-hashed password for "password123"
const HASHED_PASSWORD = bcrypt.hashSync('password123', 10);

const products = [
  {
    id: '1',
    name: 'Wireless Headphones',
    price: 79.99,
    image: '/images/headphones.jpg',
    category: 'electronics',
    stock: 50,
    description: 'Premium wireless headphones with noise cancellation',
  },
  {
    id: '2',
    name: 'Running Shoes',
    price: 129.99,
    image: '/images/shoes.jpg',
    category: 'sports',
    stock: 30,
    description: 'Lightweight running shoes for marathon training',
  },
  {
    id: '3',
    name: 'Mechanical Keyboard',
    price: 149.99,
    image: '/images/keyboard.jpg',
    category: 'electronics',
    stock: 25,
    description: 'RGB mechanical keyboard with Cherry MX switches',
  },
  {
    id: '4',
    name: 'Yoga Mat',
    price: 39.99,
    image: '/images/yoga-mat.jpg',
    category: 'sports',
    stock: 100,
    description: 'Non-slip yoga mat with carrying strap',
  },
  {
    id: '5',
    name: 'Cotton T-Shirt',
    price: 24.99,
    image: '/images/tshirt.jpg',
    category: 'clothing',
    stock: 200,
    description: 'Soft organic cotton t-shirt in multiple colors',
  },
  {
    id: '6',
    name: 'Denim Jacket',
    price: 89.99,
    image: '/images/jacket.jpg',
    category: 'clothing',
    stock: 40,
    description: 'Classic denim jacket with modern fit',
  },
  {
    id: '7',
    name: 'JavaScript: The Good Parts',
    price: 29.99,
    image: '/images/js-book.jpg',
    category: 'books',
    stock: 75,
    description: 'Essential guide to JavaScript best practices',
  },
  {
    id: '8',
    name: 'Design Patterns',
    price: 44.99,
    image: '/images/patterns-book.jpg',
    category: 'books',
    stock: 60,
    description: 'Gang of Four design patterns handbook',
  },
  {
    id: '9',
    name: 'Smart LED Bulbs (4-pack)',
    price: 34.99,
    image: '/images/led-bulbs.jpg',
    category: 'home',
    stock: 150,
    description: 'WiFi-enabled color-changing LED bulbs',
  },
  {
    id: '10',
    name: 'French Press Coffee Maker',
    price: 27.99,
    image: '/images/french-press.jpg',
    category: 'home',
    stock: 80,
    description: 'Stainless steel French press for rich coffee',
  },
];

const users = [
  {
    id: '1',
    name: 'Test User',
    email: 'test@example.com',
    password: HASHED_PASSWORD,
    role: 'user',
  },
  {
    id: '2',
    name: 'Admin User',
    email: 'admin@example.com',
    password: HASHED_PASSWORD,
    role: 'admin',
  },
];

// userId -> [{ productId, quantity }]
const carts = {};

// { id, userId, items, total, status, createdAt }
const orders = [];

// Auto-increment counters
let nextUserId = 3;
let nextOrderId = 1;

module.exports = {
  products,
  users,
  carts,
  orders,
  getNextUserId: () => String(nextUserId++),
  getNextOrderId: () => String(nextOrderId++),
};
