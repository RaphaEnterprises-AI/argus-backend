-- Migration: Profile Enhancements
-- Description: Add professional fields, social links, and avatar storage for user profiles
-- Date: 2026-01-29

-- ============================================================================
-- Add professional and social fields to user_profiles
-- ============================================================================

ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS job_title TEXT;
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS company TEXT;
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS department TEXT;
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS phone TEXT;
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS github_username TEXT;
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS linkedin_url TEXT;
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS twitter_handle TEXT;
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS website_url TEXT;

-- Add constraints for field lengths
COMMENT ON COLUMN user_profiles.job_title IS 'User job title, max 100 chars';
COMMENT ON COLUMN user_profiles.company IS 'User company name, max 100 chars';
COMMENT ON COLUMN user_profiles.department IS 'User department, max 100 chars';
COMMENT ON COLUMN user_profiles.phone IS 'User phone number, max 20 chars';
COMMENT ON COLUMN user_profiles.github_username IS 'GitHub username, max 50 chars';
COMMENT ON COLUMN user_profiles.linkedin_url IS 'LinkedIn profile URL, max 200 chars';
COMMENT ON COLUMN user_profiles.twitter_handle IS 'Twitter/X handle without @, max 50 chars';
COMMENT ON COLUMN user_profiles.website_url IS 'Personal website URL, max 200 chars';

-- ============================================================================
-- Create avatars storage bucket
-- ============================================================================

-- Create the avatars bucket (public for easy access)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'avatars',
    'avatars',
    true,
    5242880, -- 5MB limit
    ARRAY['image/jpeg', 'image/png', 'image/webp', 'image/gif']
)
ON CONFLICT (id) DO UPDATE SET
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

-- ============================================================================
-- RLS Policies for avatar storage
-- ============================================================================

-- Policy: Users can upload their own avatars (files in their user_id folder)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'objects'
        AND policyname = 'Users can upload own avatar'
    ) THEN
        CREATE POLICY "Users can upload own avatar"
        ON storage.objects FOR INSERT
        WITH CHECK (
            bucket_id = 'avatars'
            AND (storage.foldername(name))[1] = auth.uid()::text
        );
    END IF;
END $$;

-- Policy: Users can update their own avatars
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'objects'
        AND policyname = 'Users can update own avatar'
    ) THEN
        CREATE POLICY "Users can update own avatar"
        ON storage.objects FOR UPDATE
        USING (
            bucket_id = 'avatars'
            AND (storage.foldername(name))[1] = auth.uid()::text
        );
    END IF;
END $$;

-- Policy: Users can delete their own avatars
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'objects'
        AND policyname = 'Users can delete own avatar'
    ) THEN
        CREATE POLICY "Users can delete own avatar"
        ON storage.objects FOR DELETE
        USING (
            bucket_id = 'avatars'
            AND (storage.foldername(name))[1] = auth.uid()::text
        );
    END IF;
END $$;

-- Policy: Anyone can view avatars (public bucket)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'objects'
        AND policyname = 'Anyone can view avatars'
    ) THEN
        CREATE POLICY "Anyone can view avatars"
        ON storage.objects FOR SELECT
        USING (bucket_id = 'avatars');
    END IF;
END $$;

-- ============================================================================
-- Add indexes for new fields (for search/filtering)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_user_profiles_company ON user_profiles(company) WHERE company IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_profiles_job_title ON user_profiles(job_title) WHERE job_title IS NOT NULL;
