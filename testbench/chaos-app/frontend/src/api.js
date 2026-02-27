const API_BASE = '/api';

function getToken() {
  return localStorage.getItem('token');
}

function setToken(token) {
  localStorage.setItem('token', token);
}

function clearToken() {
  localStorage.removeItem('token');
}

function getUser() {
  const raw = localStorage.getItem('user');
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function setUser(user) {
  localStorage.setItem('user', JSON.stringify(user));
}

function clearUser() {
  localStorage.removeItem('user');
}

async function request(path, options = {}) {
  const token = getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (res.status === 401) {
    clearToken();
    clearUser();
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    const err = new Error(body.error || 'Request failed');
    err.status = res.status;
    err.body = body;
    throw err;
  }

  if (res.status === 204) return null;
  return res.json();
}

export async function fetchProducts(category, page = 1) {
  let url = `/products?page=${page}`;
  if (category) {
    url += `&category=${encodeURIComponent(category)}`;
  }
  return request(url);
}

export async function fetchProduct(id) {
  return request(`/products/${id}`);
}

export async function searchProducts(query) {
  return request(`/products/search?q=${encodeURIComponent(query)}`);
}

export async function login(email, password) {
  const data = await request('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  if (data.token) {
    setToken(data.token);
    setUser(data.user);
  }
  return data;
}

export async function register(name, email, password) {
  const data = await request('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ name, email, password }),
  });
  if (data.token) {
    setToken(data.token);
    setUser(data.user);
  }
  return data;
}

export async function getMe() {
  return request('/auth/me');
}

export async function getCart() {
  return request('/cart');
}

export async function addToCart(productId, quantity = 1) {
  return request('/cart/add', {
    method: 'POST',
    body: JSON.stringify({ productId, quantity }),
  });
}

export async function updateCartItem(productId, quantity) {
  return request(`/cart/${productId}`, {
    method: 'PUT',
    body: JSON.stringify({ quantity }),
  });
}

export async function removeFromCart(productId) {
  return request(`/cart/${productId}`, {
    method: 'DELETE',
  });
}

export async function checkout() {
  return request('/cart/checkout', {
    method: 'POST',
  });
}

export async function getOrders() {
  return request('/orders');
}

export async function getUserProfile(id) {
  return request(`/users/${id}`);
}

export async function updateUserProfile(id, data) {
  return request(`/users/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function getAdminProducts() {
  return request('/products');
}

export async function createProduct(product) {
  return request('/products', {
    method: 'POST',
    body: JSON.stringify(product),
  });
}

export async function updateProduct(id, product) {
  return request(`/products/${id}`, {
    method: 'PUT',
    body: JSON.stringify(product),
  });
}

export async function deleteProduct(id) {
  return request(`/products/${id}`, {
    method: 'DELETE',
  });
}

export { getToken, setToken, clearToken, getUser, setUser, clearUser };
