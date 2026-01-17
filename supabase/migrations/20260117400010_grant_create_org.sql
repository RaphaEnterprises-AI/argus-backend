-- Grant execute permission for create organization function
GRANT EXECUTE ON FUNCTION create_organization_atomic(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB) TO service_role;
