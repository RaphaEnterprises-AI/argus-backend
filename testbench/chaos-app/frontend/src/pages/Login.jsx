import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { login } from '../api';

export default function Login({ onAuthChange }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await login(email, password);
      if (onAuthChange) onAuthChange(data.user);
      navigate('/');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main role="main" aria-label="Login" className="page-container auth-page">
      <h1>Login</h1>

      <form onSubmit={handleSubmit} className="auth-form">
        {error && <p className="error-text" role="alert">{error}</p>}

        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            name="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
          />
        </div>

        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            name="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="current-password"
          />
        </div>

        <button id="login-btn" type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Logging in...' : 'Login'}
        </button>
      </form>

      <p className="auth-switch">
        Don't have an account? <Link to="/register">Register</Link>
      </p>

      <div className="auth-hint">
        <p><strong>Test credentials:</strong></p>
        <p>User: test@example.com / password123</p>
        <p>Admin: admin@example.com / password123</p>
      </div>
    </main>
  );
}
