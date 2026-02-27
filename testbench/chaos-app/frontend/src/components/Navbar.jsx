import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

export default function Navbar({ user, cartCount, onLogout }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
    navigate('/');
  };

  return (
    <nav role="navigation" aria-label="Main Navigation" className="navbar">
      <div className="navbar-brand">
        <Link to="/" tabIndex={0} className="navbar-logo">
          Chaos Shop
        </Link>
      </div>

      <div className="navbar-links">
        <Link to="/" tabIndex={0} className="nav-link">
          Home
        </Link>
        <Link to="/search" tabIndex={0} className="nav-link">
          Search
        </Link>
        <Link to="/cart" tabIndex={0} className="nav-link nav-cart">
          Cart
          {cartCount > 0 && (
            <span className="cart-badge" aria-label={`${cartCount} items in cart`}>
              {cartCount}
            </span>
          )}
        </Link>
        {user ? (
          <>
            <Link to="/profile" tabIndex={0} className="nav-link">
              {user.name || 'Profile'}
            </Link>
            {user.role === 'admin' && (
              <Link to="/admin" tabIndex={0} className="nav-link">
                Admin
              </Link>
            )}
            <button
              type="button"
              className="nav-link nav-logout-btn"
              onClick={handleLogout}
              tabIndex={0}
            >
              Logout
            </button>
          </>
        ) : (
          <Link to="/login" tabIndex={0} className="nav-link">
            Login
          </Link>
        )}
      </div>
    </nav>
  );
}
