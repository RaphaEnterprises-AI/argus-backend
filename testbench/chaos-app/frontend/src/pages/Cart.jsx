import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CartItem from '../components/CartItem';
import { getCart, updateCartItem, removeFromCart, getToken } from '../api';

export default function Cart({ onCartChange }) {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!getToken()) {
      navigate('/login');
      return;
    }
    loadCart();
  }, [navigate]);

  async function loadCart() {
    setLoading(true);
    setError(null);
    try {
      const data = await getCart();
      setItems(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const handleUpdateQuantity = async (productId, quantity) => {
    try {
      const data = await updateCartItem(productId, quantity);
      setItems(data.items || []);
      setTotal(data.total || 0);
      if (onCartChange) onCartChange();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRemove = async (productId) => {
    try {
      const data = await removeFromCart(productId);
      setItems(data.items || []);
      setTotal(data.total || 0);
      if (onCartChange) onCartChange();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleCheckout = () => {
    navigate('/checkout');
  };

  if (loading) {
    return (
      <main role="main" aria-label="Shopping Cart" className="page-container">
        <h1>Shopping Cart</h1>
        <p className="loading-text">Loading cart...</p>
      </main>
    );
  }

  return (
    <main role="main" aria-label="Shopping Cart" className="page-container">
      <h1>Shopping Cart</h1>

      {error && <p className="error-text" role="alert">{error}</p>}

      {items.length === 0 ? (
        <p>Your cart is empty.</p>
      ) : (
        <>
          <div className="cart-items">
            {items.map((item) => (
              <CartItem
                key={item.productId}
                item={item}
                onUpdateQuantity={handleUpdateQuantity}
                onRemove={handleRemove}
              />
            ))}
          </div>

          <div className="cart-summary">
            <p className="cart-total">
              Total: <strong>${total.toFixed(2)}</strong>
            </p>
            <button
              id="checkout-btn"
              className="checkout-submit"
              onClick={handleCheckout}
              type="button"
            >
              Proceed to Checkout
            </button>
          </div>
        </>
      )}
    </main>
  );
}
