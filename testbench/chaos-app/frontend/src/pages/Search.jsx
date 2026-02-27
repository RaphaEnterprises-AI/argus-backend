import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import ProductCard from '../components/ProductCard';
import { searchProducts, addToCart, getToken } from '../api';

export default function Search({ onCartChange }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get('q') || '');
  const [results, setResults] = useState([]);
  const [searchedQuery, setSearchedQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const q = searchParams.get('q');
    if (q) {
      setQuery(q);
      performSearch(q);
    }
  }, [searchParams]);

  async function performSearch(q) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await searchProducts(q);
      setResults(data.results || []);
      setSearchedQuery(q);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      setSearchParams({ q: query });
    }
  };

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
    <main role="main" aria-label="Search Products" className="page-container">
      <h1>Search Products</h1>

      <form onSubmit={handleSubmit} className="search-form" role="search">
        <label htmlFor="search-input" className="sr-only">Search</label>
        <input
          id="search-input"
          name="q"
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for products..."
          autoComplete="off"
        />
        <button type="submit" className="btn-primary">
          Search
        </button>
      </form>

      {loading && <p className="loading-text">Searching...</p>}
      {error && <p className="error-text" role="alert">{error}</p>}

      {searchedQuery && !loading && (
        <>
          <p>Results for: {searchedQuery}</p>
          {results.length === 0 ? (
            <p>No products found matching your search.</p>
          ) : (
            <div className="product-grid">
              {results.map((product) => (
                <ProductCard
                  key={product.id}
                  product={product}
                  onAddToCart={handleAddToCart}
                />
              ))}
            </div>
          )}
        </>
      )}
    </main>
  );
}
