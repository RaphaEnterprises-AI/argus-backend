import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCart, checkout as apiCheckout, getToken } from '../api';

export default function Checkout({ onCartChange }) {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [orderConfirmed, setOrderConfirmed] = useState(null);

  // Shipping form state
  const [shipping, setShipping] = useState({
    firstName: '',
    lastName: '',
    address: '',
    city: '',
    state: '',
    zip: '',
  });

  // Payment form state
  const [payment, setPayment] = useState({
    cardNumber: '',
    cardName: '',
    expiry: '',
    cvv: '',
  });

  useEffect(() => {
    if (!getToken()) {
      navigate('/login');
      return;
    }
    loadCart();
  }, [navigate]);

  async function loadCart() {
    setLoading(true);
    try {
      const data = await getCart();
      if (!data.items || data.items.length === 0) {
        navigate('/cart');
        return;
      }
      setItems(data.items);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const handleShippingChange = (e) => {
    setShipping({ ...shipping, [e.target.name]: e.target.value });
  };

  const handlePaymentChange = (e) => {
    setPayment({ ...payment, [e.target.name]: e.target.value });
  };

  const handleShippingSubmit = (e) => {
    e.preventDefault();
    setStep(2);
  };

  const handlePaymentSubmit = (e) => {
    e.preventDefault();
    setStep(3);
  };

  const handlePlaceOrder = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const order = await apiCheckout();
      setOrderConfirmed(order);
      if (onCartChange) onCartChange();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <main role="main" aria-label="Checkout" className="page-container">
        <h1>Checkout</h1>
        <p className="loading-text">Loading...</p>
      </main>
    );
  }

  if (orderConfirmed) {
    return (
      <main role="main" aria-label="Order Confirmation" className="page-container">
        <h1>Order Confirmed</h1>
        <div className="order-confirmation">
          <p>Thank you for your order!</p>
          <p>Order ID: <strong>{orderConfirmed.id}</strong></p>
          <p>Total: <strong>${orderConfirmed.total.toFixed(2)}</strong></p>
          <p>Status: <strong>{orderConfirmed.status}</strong></p>
          <button type="button" onClick={() => navigate('/')}>
            Continue Shopping
          </button>
        </div>
      </main>
    );
  }

  return (
    <main role="main" aria-label="Checkout" className="page-container">
      <h1>Checkout</h1>

      {error && <p className="error-text" role="alert">{error}</p>}

      <div className="checkout-steps">
        <span className={`checkout-step ${step >= 1 ? 'active' : ''}`}>1. Shipping</span>
        <span className={`checkout-step ${step >= 2 ? 'active' : ''}`}>2. Payment</span>
        <span className={`checkout-step ${step >= 3 ? 'active' : ''}`}>3. Confirm</span>
      </div>

      {step === 1 && (
        <div className="checkout-form-container">
          <h2>Shipping Address</h2>
          <form onSubmit={handleShippingSubmit}>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="firstName">First Name</label>
                <input
                  id="firstName"
                  name="firstName"
                  type="text"
                  value={shipping.firstName}
                  onChange={handleShippingChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="lastName">Last Name</label>
                <input
                  id="lastName"
                  name="lastName"
                  type="text"
                  value={shipping.lastName}
                  onChange={handleShippingChange}
                  required
                />
              </div>
            </div>
            <div className="form-group">
              <label htmlFor="address">Street Address</label>
              <input
                id="address"
                name="address"
                type="text"
                value={shipping.address}
                onChange={handleShippingChange}
                required
              />
            </div>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="city">City</label>
                <input
                  id="city"
                  name="city"
                  type="text"
                  value={shipping.city}
                  onChange={handleShippingChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="state">State</label>
                <input
                  id="state"
                  name="state"
                  type="text"
                  value={shipping.state}
                  onChange={handleShippingChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="zip">ZIP Code</label>
                <input
                  id="zip"
                  name="zip"
                  type="text"
                  value={shipping.zip}
                  onChange={handleShippingChange}
                  required
                />
              </div>
            </div>
            <div className="form-actions">
              <button type="submit" className="btn-primary">
                Continue to Payment
              </button>
            </div>
          </form>
        </div>
      )}

      {step === 2 && (
        <div className="checkout-form-container">
          <h2>Payment Information</h2>
          <form onSubmit={handlePaymentSubmit}>
            <div className="form-group">
              <label htmlFor="cardName">Name on Card</label>
              <input
                id="cardName"
                name="cardName"
                type="text"
                value={payment.cardName}
                onChange={handlePaymentChange}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="cardNumber">Card Number</label>
              <input
                id="cardNumber"
                name="cardNumber"
                type="text"
                value={payment.cardNumber}
                onChange={handlePaymentChange}
                placeholder="1234 5678 9012 3456"
                required
              />
            </div>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="expiry">Expiry Date</label>
                <input
                  id="expiry"
                  name="expiry"
                  type="text"
                  value={payment.expiry}
                  onChange={handlePaymentChange}
                  placeholder="MM/YY"
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="cvv">CVV</label>
                <input
                  id="cvv"
                  name="cvv"
                  type="text"
                  value={payment.cvv}
                  onChange={handlePaymentChange}
                  placeholder="123"
                  required
                />
              </div>
            </div>
            <div className="form-actions">
              <button type="button" className="btn-secondary" onClick={() => setStep(1)}>
                Back
              </button>
              <button type="submit" className="btn-primary">
                Review Order
              </button>
            </div>
          </form>
        </div>
      )}

      {step === 3 && (
        <div className="checkout-form-container">
          <h2>Review &amp; Confirm</h2>

          <div className="checkout-review">
            <h3>Shipping</h3>
            <p>
              {shipping.firstName} {shipping.lastName}<br />
              {shipping.address}<br />
              {shipping.city}, {shipping.state} {shipping.zip}
            </p>

            <h3>Payment</h3>
            <p>Card ending in {payment.cardNumber.slice(-4) || '****'}</p>

            <h3>Items ({items.length})</h3>
            <ul className="checkout-items-list">
              {items.map((item) => (
                <li key={item.productId}>
                  {item.name} x {item.quantity} &mdash; ${(item.price * item.quantity).toFixed(2)}
                </li>
              ))}
            </ul>

            <p className="checkout-total">
              Total: <strong>${total.toFixed(2)}</strong>
            </p>
          </div>

          <div className="checkout-actions">
            <section>
              <div>
                <button type="button" className="btn-secondary" onClick={() => setStep(2)}>
                  Back
                </button>
                <button
                  id="place-order-btn"
                  type="button"
                  className="btn-primary"
                  onClick={handlePlaceOrder}
                  disabled={submitting}
                >
                  {submitting ? 'Placing Order...' : 'Place Order'}
                </button>
              </div>
            </section>
          </div>
        </div>
      )}
    </main>
  );
}
