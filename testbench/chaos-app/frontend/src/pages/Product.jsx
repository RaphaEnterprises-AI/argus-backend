import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { fetchProduct, addToCart, getToken } from '../api';

export default function Product({ onCartChange }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const [product, setProduct] = useState(null);
  const [quantity, setQuantity] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [addedMessage, setAddedMessage] = useState('');

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchProduct(id);
        setProduct(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  const handleAddToCart = async () => {
    if (!getToken()) {
      navigate('/login');
      return;
    }
    try {
      await addToCart(product.id, quantity);
      setAddedMessage(`Added ${quantity} x ${product.name} to cart`);
      if (onCartChange) onCartChange();
      setTimeout(() => setAddedMessage(''), 3000);
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <main role="main" className="page-container">
        <p className="loading-text">Loading product...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main role="main" className="page-container">
        <p className="error-text" role="alert">{error}</p>
      </main>
    );
  }

  if (!product) {
    return (
      <main role="main" className="page-container">
        <p>Product not found.</p>
      </main>
    );
  }

  return (
    <main role="main" className="page-container">
      <div className="product-detail">
        <div className="product-detail-image-container">
          <img
            src={product.image}
            alt={product.name}
            className="product-detail-image"
          />
        </div>
        <div className="product-detail-info">
          <h1 className="product-detail-name">{product.name}</h1>
          <p className="product-detail-category">{product.category}</p>
          <p className="product-detail-price">${product.price.toFixed(2)}</p>
          <p className="product-detail-description">{product.description}</p>
          <p className="product-detail-stock">
            {product.stock > 0 ? `${product.stock} in stock` : 'Out of stock'}
          </p>

          <div className="product-detail-actions">
            <label htmlFor="quantity-select">Quantity:</label>
            <select
              id="quantity-select"
              value={quantity}
              onChange={(e) => setQuantity(parseInt(e.target.value, 10))}
            >
              {Array.from({ length: Math.min(product.stock, 10) }, (_, i) => (
                <option key={i + 1} value={i + 1}>
                  {i + 1}
                </option>
              ))}
            </select>

            <button
              className="add-to-cart"
              id="add-to-cart-btn"
              onClick={handleAddToCart}
              disabled={product.stock < 1}
              type="button"
            >
              Add to Cart
            </button>
          </div>

          {addedMessage && (
            <p className="success-text" role="status">{addedMessage}</p>
          )}
        </div>
      </div>
    </main>
  );
}
