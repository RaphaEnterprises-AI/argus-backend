import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  getAdminProducts,
  createProduct,
  updateProduct,
  deleteProduct,
  getToken,
} from '../api';

export default function Admin({ user }) {
  const navigate = useNavigate();
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState({
    name: '',
    price: '',
    category: '',
    stock: '',
    description: '',
    image: '',
  });

  useEffect(() => {
    if (!getToken()) {
      navigate('/login');
      return;
    }
    if (!user || user.role !== 'admin') {
      navigate('/');
      return;
    }
    loadProducts();
  }, [navigate, user]);

  async function loadProducts() {
    setLoading(true);
    try {
      const data = await getAdminProducts();
      setProducts(data.products || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const handleFormChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleAdd = () => {
    setEditingId(null);
    setForm({
      name: '',
      price: '',
      category: '',
      stock: '',
      description: '',
      image: '',
    });
    setShowForm(true);
  };

  const handleEdit = (product) => {
    setEditingId(product.id);
    setForm({
      name: product.name,
      price: String(product.price),
      category: product.category,
      stock: String(product.stock),
      description: product.description,
      image: product.image,
    });
    setShowForm(true);
  };

  const handleDelete = async (id) => {
    if (!confirm('Delete this product?')) return;
    try {
      await deleteProduct(id);
      setProducts(products.filter((p) => p.id !== id));
    } catch (err) {
      setError(err.message);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    const payload = {
      name: form.name,
      price: parseFloat(form.price),
      category: form.category,
      stock: parseInt(form.stock, 10) || 0,
      description: form.description,
      image: form.image || '/images/placeholder.jpg',
    };

    try {
      if (editingId) {
        const updated = await updateProduct(editingId, payload);
        setProducts(products.map((p) => (p.id === editingId ? updated : p)));
      } else {
        const created = await createProduct(payload);
        setProducts([...products, created]);
      }
      setShowForm(false);
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <main role="main" aria-label="Admin Panel" className="page-container">
        <h1>Admin Panel</h1>
        <p className="loading-text">Loading products...</p>
      </main>
    );
  }

  return (
    <main role="main" aria-label="Admin Panel" className="page-container">
      <h1>Admin Panel</h1>

      {error && <p className="error-text" role="alert">{error}</p>}

      <div className="admin-toolbar">
        <button type="button" className="btn-primary" onClick={handleAdd}>
          Add Product
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleSubmit} className="admin-form">
          <h2>{editingId ? 'Edit Product' : 'New Product'}</h2>
          <div className="form-group">
            <label htmlFor="admin-name">Name</label>
            <input
              id="admin-name"
              name="name"
              type="text"
              value={form.name}
              onChange={handleFormChange}
              required
            />
          </div>
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="admin-price">Price</label>
              <input
                id="admin-price"
                name="price"
                type="number"
                step="0.01"
                value={form.price}
                onChange={handleFormChange}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="admin-category">Category</label>
              <input
                id="admin-category"
                name="category"
                type="text"
                value={form.category}
                onChange={handleFormChange}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="admin-stock">Stock</label>
              <input
                id="admin-stock"
                name="stock"
                type="number"
                value={form.stock}
                onChange={handleFormChange}
              />
            </div>
          </div>
          <div className="form-group">
            <label htmlFor="admin-description">Description</label>
            <textarea
              id="admin-description"
              name="description"
              value={form.description}
              onChange={handleFormChange}
              rows={3}
            />
          </div>
          <div className="form-group">
            <label htmlFor="admin-image">Image URL</label>
            <input
              id="admin-image"
              name="image"
              type="text"
              value={form.image}
              onChange={handleFormChange}
              placeholder="/images/placeholder.jpg"
            />
          </div>
          <div className="form-actions">
            <button type="submit" className="btn-primary">
              {editingId ? 'Update' : 'Create'}
            </button>
            <button type="button" className="btn-secondary" onClick={() => setShowForm(false)}>
              Cancel
            </button>
          </div>
        </form>
      )}

      <table className="admin-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Category</th>
            <th>Price</th>
            <th>Stock</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {products.map((product) => (
            <tr key={product.id}>
              <td>{product.id}</td>
              <td>{product.name}</td>
              <td>{product.category}</td>
              <td>${typeof product.price === 'number' ? product.price.toFixed(2) : product.price}</td>
              <td>{product.stock}</td>
              <td>
                <button
                  type="button"
                  className="btn-small"
                  onClick={() => handleEdit(product)}
                >
                  Edit
                </button>
                <button
                  type="button"
                  className="btn-small btn-danger"
                  onClick={() => handleDelete(product.id)}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
