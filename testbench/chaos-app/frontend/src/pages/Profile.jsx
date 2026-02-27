import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getMe, getOrders, updateUserProfile, getToken, setUser } from '../api';

export default function Profile({ user }) {
  const navigate = useNavigate();
  const [profile, setProfile] = useState(null);
  const [orders, setOrders] = useState([]);
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editEmail, setEditEmail] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    if (!getToken()) {
      navigate('/login');
      return;
    }
    loadProfile();
    loadOrders();
  }, [navigate]);

  async function loadProfile() {
    setLoading(true);
    try {
      const data = await getMe();
      setProfile(data);
      setEditName(data.name);
      setEditEmail(data.email);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadOrders() {
    try {
      const data = await getOrders();
      setOrders(data.orders || []);
    } catch {
      // Silently fail for orders
    }
  }

  const handleEdit = () => {
    setEditing(true);
    setSuccess(null);
  };

  const handleCancel = () => {
    setEditing(false);
    setEditName(profile.name);
    setEditEmail(profile.email);
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const updated = await updateUserProfile(profile.id, {
        name: editName,
        email: editEmail,
      });
      setProfile(updated);
      setUser(updated);
      setEditing(false);
      setSuccess('Profile updated successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <main role="main" aria-label="User Profile" className="page-container">
        <h1>Profile</h1>
        <p className="loading-text">Loading profile...</p>
      </main>
    );
  }

  return (
    <main role="main" aria-label="User Profile" className="page-container">
      <h1>Profile</h1>

      {error && <p className="error-text" role="alert">{error}</p>}
      {success && <p className="success-text" role="status">{success}</p>}

      {profile && (
        <div className="profile-section">
          {editing ? (
            <form onSubmit={handleSave} className="profile-form">
              <div className="form-group">
                <label htmlFor="edit-name">Name</label>
                <input
                  id="edit-name"
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="edit-email">Email</label>
                <input
                  id="edit-email"
                  type="email"
                  value={editEmail}
                  onChange={(e) => setEditEmail(e.target.value)}
                  required
                />
              </div>
              <div className="form-actions">
                <button type="submit" className="btn-primary">Save</button>
                <button type="button" className="btn-secondary" onClick={handleCancel}>Cancel</button>
              </div>
            </form>
          ) : (
            <div className="profile-details">
              <p><strong>Name:</strong> {profile.name}</p>
              <p><strong>Email:</strong> {profile.email}</p>
              <p><strong>Role:</strong> {profile.role}</p>
              <button type="button" className="btn-primary" onClick={handleEdit}>
                Edit Profile
              </button>
            </div>
          )}
        </div>
      )}

      <div className="orders-section">
        <h2>Order History</h2>
        {orders.length === 0 ? (
          <p>No orders yet.</p>
        ) : (
          <table className="orders-table">
            <thead>
              <tr>
                <th>Order ID</th>
                <th>Date</th>
                <th>Items</th>
                <th>Total</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((order) => (
                <tr key={order.id}>
                  <td>{order.id}</td>
                  <td>{new Date(order.createdAt).toLocaleDateString()}</td>
                  <td>{order.items.length}</td>
                  <td>${order.total.toFixed(2)}</td>
                  <td>{order.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </main>
  );
}
