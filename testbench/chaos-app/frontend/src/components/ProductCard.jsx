import React from 'react';
import { Link } from 'react-router-dom';

export default function ProductCard({ product, onAddToCart }) {
  const handleAdd = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (onAddToCart) {
      onAddToCart(product.id);
    }
  };

  return (
    <div className="product-card">
      <Link to={`/product/${product.id}`} className="product-card-link">
        <img
          src={product.image}
          alt={product.name}
          className="product-card-image"
          loading="lazy"
        />
        <div className="product-card-info">
          <h3 className="product-card-name">{product.name}</h3>
          <p className="product-card-category">{product.category}</p>
          <p className="product-card-price">${product.price.toFixed(2)}</p>
        </div>
      </Link>
      <button className="add-to-cart" onClick={handleAdd} type="button">
        Add to Cart
      </button>
    </div>
  );
}
