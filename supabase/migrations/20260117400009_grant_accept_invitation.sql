-- Grant execute permission for accept invitation function
GRANT EXECUTE ON FUNCTION accept_invitation_atomic(UUID, TEXT, TEXT) TO service_role;
