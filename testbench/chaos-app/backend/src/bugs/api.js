const { isBugEnabled } = require('./registry');

/**
 * Deep clone data to avoid mutating the original db objects.
 */
function cloneData(data) {
  return JSON.parse(JSON.stringify(data));
}

/**
 * Transform JSON response data for API contract bugs.
 * Always works on a clone to avoid mutating the in-memory store.
 */
function transformJson(req, res, data) {
  let needsClone = false;

  if (isBugEnabled('api-schema-drift') || isBugEnabled('api-missing-field')) {
    needsClone = true;
  }

  if (needsClone && data && typeof data === 'object') {
    data = cloneData(data);
  }

  // api-schema-drift: Convert price from number to string
  if (isBugEnabled('api-schema-drift')) {
    if (Array.isArray(data)) {
      data.forEach(convertPriceToString);
    } else if (data && typeof data === 'object') {
      convertPriceToString(data);
      // Also handle nested items arrays (e.g., order items)
      if (Array.isArray(data.products)) {
        data.products.forEach(convertPriceToString);
      }
      if (Array.isArray(data.items)) {
        data.items.forEach(convertPriceToString);
      }
    }
  }

  // api-missing-field: Remove email from user responses
  if (isBugEnabled('api-missing-field')) {
    if (Array.isArray(data)) {
      data.forEach(removeEmail);
    } else if (data && typeof data === 'object') {
      removeEmail(data);
      if (data.user) {
        removeEmail(data.user);
      }
    }
  }

  return data;
}

function convertPriceToString(obj) {
  if (obj && typeof obj.price === 'number') {
    obj.price = String(obj.price);
  }
}

function removeEmail(obj) {
  if (obj && 'email' in obj) {
    delete obj.email;
  }
}

/**
 * Intercept response status codes for API bugs.
 * Must be called before route handlers.
 */
function interceptStatus(req, res) {
  // api-wrong-status: Override 201 to 200
  if (isBugEnabled('api-wrong-status')) {
    const originalStatus = res.status.bind(res);
    res.status = function (code) {
      if (code === 201) {
        return originalStatus(200);
      }
      return originalStatus(code);
    };
  }

  // api-500-random: 10% chance of random 500
  if (isBugEnabled('api-500-random') && req.path.startsWith('/api/')) {
    if (Math.random() < 0.1) {
      res._chaos500 = true;
    }
  }
}

/**
 * Check and apply random 500 error.
 * Returns true if request was terminated.
 */
function checkRandom500(req, res) {
  if (res._chaos500) {
    res.status(500).json({
      error: 'Internal Server Error',
      message: 'An unexpected error occurred',
      chaos: true,
    });
    return true;
  }
  return false;
}

/**
 * Transform pagination parameters.
 * When api-breaking-pagination is enabled, ignore ?page= and only accept ?offset=
 */
function transformPagination(req) {
  if (isBugEnabled('api-breaking-pagination')) {
    // If client sends ?page=, ignore it
    if (req.query.page && !req.query.offset) {
      delete req.query.page;
      // Don't apply any pagination — client must use ?offset= instead
      req._paginationBroken = true;
    }
  }
}

module.exports = {
  transformJson,
  interceptStatus,
  checkRandom500,
  transformPagination,
};
