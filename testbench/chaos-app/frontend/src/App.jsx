import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Product from './pages/Product';
import Cart from './pages/Cart';
import Checkout from './pages/Checkout';
import Login from './pages/Login';
import Register from './pages/Register';
import Profile from './pages/Profile';
import Search from './pages/Search';
import Admin from './pages/Admin';
import { getUser, getCart as fetchCart, getToken } from './api';

export default function App() {
  const [user, setUser] = useState(() => getUser());
  const [cartCount, setCartCount] = useState(0);

  const refreshCart = useCallback(async () => {
    if (!getToken()) {
      setCartCount(0);
      return;
    }
    try {
      const data = await fetchCart();
      const count = (data.items || []).reduce((sum, item) => sum + item.quantity, 0);
      setCartCount(count);
    } catch {
      setCartCount(0);
    }
  }, []);

  useEffect(() => {
    if (user) {
      refreshCart();
    } else {
      setCartCount(0);
    }
  }, [user, refreshCart]);

  const handleAuthChange = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setCartCount(0);
  };

  return (
    <BrowserRouter>
      <Navbar user={user} cartCount={cartCount} onLogout={handleLogout} />
      <Routes>
        <Route path="/" element={<Home onCartChange={refreshCart} />} />
        <Route path="/product/:id" element={<Product onCartChange={refreshCart} />} />
        <Route path="/cart" element={<Cart onCartChange={refreshCart} />} />
        <Route path="/checkout" element={<Checkout onCartChange={refreshCart} />} />
        <Route path="/login" element={<Login onAuthChange={handleAuthChange} />} />
        <Route path="/register" element={<Register onAuthChange={handleAuthChange} />} />
        <Route path="/profile" element={<Profile user={user} />} />
        <Route path="/search" element={<Search onCartChange={refreshCart} />} />
        <Route path="/admin" element={<Admin user={user} />} />
      </Routes>
    </BrowserRouter>
  );
}
