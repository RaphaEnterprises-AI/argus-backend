import React, { useState, useEffect } from 'react';
import ProductCard from '../components/ProductCard';
import { fetchProducts, addToCart, getToken } from '../api';

const CATEGORIES = ['all', 'electronics', 'sports', 'clothing', 'books', 'home'];

export default function Home({ onCartChange }) {
  const [products, setProducts] = useState([]);
  const [category, setCategory] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const cat = category === 'all' ? undefined : category;
        const data = await fetchProducts(cat);
        setProducts(data.products || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [category]);

  const handleAddToCart = async (productId) => {
    if (!getToken()) {
      window.location.href = '/login';
      return;
    }
    try {
      await addToCart(productId, 1);
      if (onCartChange) onCartChange();
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <main role="main" aria-label="Product Listing" className="page-container">
      <h1>View All Products</h1>

      <div className="filter-bar">
        <label htmlFor="category-filter">Filter by Category:</label>
        <select
          id="category-filter"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
        >
          {CATEGORIES.map((cat) => (
            <option key={cat} value={cat}>
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {loading && <p className="loading-text">Loading products...</p>}
      {error && <p className="error-text" role="alert">{error}</p>}

      {!loading && !error && products.length === 0 && (
        <p>No products found in this category.</p>
      )}

      <div className="product-grid">
        {products.map((product) => (
          <ProductCard
            key={product.id}
            product={product}
            onAddToCart={handleAddToCart}
          />
        ))}
      </div>
    </main>
  );
}
