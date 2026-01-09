-- Fix user_profiles email to be nullable
-- Email might not always be available from JWT claims
ALTER TABLE user_profiles ALTER COLUMN email DROP NOT NULL;
