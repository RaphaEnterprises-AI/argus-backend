-- Add tool_invocations column to chat_messages
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS tool_invocations JSONB DEFAULT NULL;

-- Update message_count and preview columns if missing
ALTER TABLE chat_conversations ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0;
ALTER TABLE chat_conversations ADD COLUMN IF NOT EXISTS preview TEXT;

-- Create or replace the trigger function for message count
CREATE OR REPLACE FUNCTION update_conversation_message_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chat_conversations
    SET message_count = message_count + 1,
        updated_at = NOW(),
        preview = CASE
            WHEN NEW.role = 'user' THEN LEFT(NEW.content, 100)
            ELSE preview
        END
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop and recreate trigger if exists
DROP TRIGGER IF EXISTS update_message_count_trigger ON chat_messages;
CREATE TRIGGER update_message_count_trigger
    AFTER INSERT ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION update_conversation_message_count();

-- Create or replace the trigger function for auto-title
CREATE OR REPLACE FUNCTION set_conversation_title()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.role = 'user' THEN
        UPDATE chat_conversations
        SET title = COALESCE(title, LEFT(NEW.content, 50))
        WHERE id = NEW.conversation_id AND title IS NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_conversation_title_trigger ON chat_messages;
CREATE TRIGGER set_conversation_title_trigger
    AFTER INSERT ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION set_conversation_title();
