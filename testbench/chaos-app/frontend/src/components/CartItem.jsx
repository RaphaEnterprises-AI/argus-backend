import React from 'react';

export default function CartItem({ item, onUpdateQuantity, onRemove }) {
  const handleIncrement = () => {
    onUpdateQuantity(item.productId, item.quantity + 1);
  };

  const handleDecrement = () => {
    if (item.quantity > 1) {
      onUpdateQuantity(item.productId, item.quantity - 1);
    }
  };

  const handleRemove = () => {
    onRemove(item.productId);
  };

  return (
    <div className="cart-item" id={`item-${item.productId}`}>
      <img
        src={item.image}
        alt={item.name}
        className="cart-item-image"
      />
      <div className="cart-item-details">
        <h3 className="cart-item-name">{item.name}</h3>
        <p className="cart-item-price">${item.price.toFixed(2)}</p>
      </div>
      <div className="cart-item-quantity">
        <button
          type="button"
          className="qty-btn qty-decrement"
          onClick={handleDecrement}
          disabled={item.quantity <= 1}
          aria-label="Decrease quantity"
        >
          -
        </button>
        <span className="qty-value" aria-label={`Quantity: ${item.quantity}`}>
          {item.quantity}
        </span>
        <button
          type="button"
          className="qty-btn qty-increment"
          onClick={handleIncrement}
          aria-label="Increase quantity"
        >
          +
        </button>
      </div>
      <p className="cart-item-subtotal">
        ${(item.price * item.quantity).toFixed(2)}
      </p>
      <button
        type="button"
        className="cart-item-remove"
        onClick={handleRemove}
        aria-label={`Remove ${item.name} from cart`}
      >
        Remove
      </button>
    </div>
  );
}
