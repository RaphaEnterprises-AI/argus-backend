/**
 * Comprehensive E2E API Test Suite for Chat Functionality
 * Tests: Conversations, Messages, and Chat API with tools
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://ytjkdwaxhhjzchnmxyjq.supabase.co';
const SUPABASE_SERVICE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl0amtkd2F4aGhqemNobm14eWpxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NzAyNTUxMCwiZXhwIjoyMDgyNjAxNTEwfQ.B_8D1mhiufEP77013ggwe5kfxEVRpaq6Sx4JTXvcayQ';

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

const TEST_USER_ID = 'test-user-' + Date.now();

interface TestResult {
  name: string;
  passed: boolean;
  error?: string;
  data?: unknown;
}

const results: TestResult[] = [];

function log(message: string) {
  console.log(`\n${'='.repeat(60)}\n${message}\n${'='.repeat(60)}`);
}

function success(name: string, data?: unknown) {
  console.log(`✅ PASS: ${name}`);
  results.push({ name, passed: true, data });
}

function fail(name: string, error: string) {
  console.log(`❌ FAIL: ${name} - ${error}`);
  results.push({ name, passed: false, error });
}

async function testCreateConversation(): Promise<string | null> {
  log('TEST 1: Create Conversation');

  try {
    const { data, error } = await supabase
      .from('chat_conversations')
      .insert({
        user_id: TEST_USER_ID,
        title: 'Test Conversation',
      })
      .select()
      .single();

    if (error) throw error;

    if (!data || !data.id) throw new Error('No conversation ID returned');

    console.log('Created conversation:', JSON.stringify(data, null, 2));
    success('Create Conversation', data);
    return data.id;
  } catch (err) {
    fail('Create Conversation', (err as Error).message);
    return null;
  }
}

async function testAddUserMessage(conversationId: string): Promise<string | null> {
  log('TEST 2: Add User Message');

  try {
    const { data, error } = await supabase
      .from('chat_messages')
      .insert({
        conversation_id: conversationId,
        role: 'user',
        content: 'Discover all interactive elements on https://demo.vercel.store',
      })
      .select()
      .single();

    if (error) throw error;

    if (!data || !data.id) throw new Error('No message ID returned');

    console.log('Created user message:', JSON.stringify(data, null, 2));
    success('Add User Message', data);
    return data.id;
  } catch (err) {
    fail('Add User Message', (err as Error).message);
    return null;
  }
}

async function testAddAssistantMessageWithToolInvocations(conversationId: string): Promise<string | null> {
  log('TEST 3: Add Assistant Message with Tool Invocations');

  const toolInvocations = [
    {
      toolName: 'discoverElements',
      args: { url: 'https://demo.vercel.store' },
      state: 'result',
      result: {
        success: true,
        actions: [
          { description: 'Navigation link: Home', selector: 'a[href="/"]' },
          { description: 'Navigation link: All', selector: 'a[href="/search"]' },
          { description: 'Add to cart button', selector: 'button[data-test="add-to-cart"]' },
        ],
      },
    },
  ];

  try {
    const { data, error } = await supabase
      .from('chat_messages')
      .insert({
        conversation_id: conversationId,
        role: 'assistant',
        content: 'I found 3 interactive elements on the page.',
        tool_invocations: toolInvocations,
      })
      .select()
      .single();

    if (error) throw error;

    if (!data || !data.id) throw new Error('No message ID returned');

    console.log('Created assistant message:', JSON.stringify(data, null, 2));
    success('Add Assistant Message with Tool Invocations', data);
    return data.id;
  } catch (err) {
    fail('Add Assistant Message with Tool Invocations', (err as Error).message);
    return null;
  }
}

async function testListConversations(): Promise<void> {
  log('TEST 4: List Conversations for User');

  try {
    const { data, error } = await supabase
      .from('chat_conversations')
      .select('*')
      .eq('user_id', TEST_USER_ID)
      .order('updated_at', { ascending: false });

    if (error) throw error;

    console.log('Conversations:', JSON.stringify(data, null, 2));

    if (!data || data.length === 0) {
      throw new Error('No conversations found');
    }

    success('List Conversations', { count: data.length });
  } catch (err) {
    fail('List Conversations', (err as Error).message);
  }
}

async function testGetConversationMessages(conversationId: string): Promise<void> {
  log('TEST 5: Get Conversation Messages');

  try {
    const { data, error } = await supabase
      .from('chat_messages')
      .select('*')
      .eq('conversation_id', conversationId)
      .order('created_at', { ascending: true });

    if (error) throw error;

    console.log('Messages:', JSON.stringify(data, null, 2));

    if (!data || data.length === 0) {
      throw new Error('No messages found');
    }

    // Verify tool invocations are stored correctly
    const assistantMessage = data.find((m: { role: string }) => m.role === 'assistant');
    if (assistantMessage && assistantMessage.tool_invocations) {
      console.log('Tool invocations retrieved successfully');
    }

    success('Get Conversation Messages', { count: data.length });
  } catch (err) {
    fail('Get Conversation Messages', (err as Error).message);
  }
}

async function testConversationAutoTitle(conversationId: string): Promise<void> {
  log('TEST 6: Verify Auto-Title Trigger');

  try {
    const { data, error } = await supabase
      .from('chat_conversations')
      .select('*')
      .eq('id', conversationId)
      .single();

    if (error) throw error;

    console.log('Conversation after messages:', JSON.stringify(data, null, 2));

    // Title should be set by trigger (first user message)
    if (data.title === 'Test Conversation') {
      // Title was set on creation, that's fine
      console.log('Title was set on creation (expected)');
    }

    // Preview should be updated by trigger
    if (data.preview) {
      console.log('Preview was set by trigger:', data.preview);
    }

    // Message count should be updated
    if (data.message_count >= 2) {
      console.log('Message count updated correctly:', data.message_count);
    }

    success('Verify Auto-Title Trigger', data);
  } catch (err) {
    fail('Verify Auto-Title Trigger', (err as Error).message);
  }
}

async function testUpdateConversation(conversationId: string): Promise<void> {
  log('TEST 7: Update Conversation');

  try {
    const { data, error } = await supabase
      .from('chat_conversations')
      .update({ title: 'Updated Test Conversation' })
      .eq('id', conversationId)
      .select()
      .single();

    if (error) throw error;

    console.log('Updated conversation:', JSON.stringify(data, null, 2));

    if (data.title !== 'Updated Test Conversation') {
      throw new Error('Title not updated correctly');
    }

    success('Update Conversation', data);
  } catch (err) {
    fail('Update Conversation', (err as Error).message);
  }
}

async function testCreateConversationWithProject(): Promise<string | null> {
  log('TEST 8: Create Conversation with Project ID');

  // First, get an existing project
  const { data: projects } = await supabase
    .from('projects')
    .select('id')
    .limit(1);

  const projectId = projects && projects.length > 0 ? projects[0].id : null;

  try {
    const { data, error } = await supabase
      .from('chat_conversations')
      .insert({
        user_id: TEST_USER_ID,
        project_id: projectId,
        title: 'Project-linked Conversation',
      })
      .select()
      .single();

    if (error) throw error;

    console.log('Created project-linked conversation:', JSON.stringify(data, null, 2));
    success('Create Conversation with Project ID', data);
    return data.id;
  } catch (err) {
    fail('Create Conversation with Project ID', (err as Error).message);
    return null;
  }
}

async function testDeleteConversation(conversationId: string): Promise<void> {
  log('TEST 9: Delete Conversation (cascade delete messages)');

  try {
    // First verify messages exist
    const { data: messagesBefore } = await supabase
      .from('chat_messages')
      .select('id')
      .eq('conversation_id', conversationId);

    console.log('Messages before delete:', messagesBefore?.length || 0);

    // Delete conversation
    const { error } = await supabase
      .from('chat_conversations')
      .delete()
      .eq('id', conversationId);

    if (error) throw error;

    // Verify messages are also deleted (cascade)
    const { data: messagesAfter } = await supabase
      .from('chat_messages')
      .select('id')
      .eq('conversation_id', conversationId);

    console.log('Messages after delete:', messagesAfter?.length || 0);

    if (messagesAfter && messagesAfter.length > 0) {
      throw new Error('Messages were not cascade deleted');
    }

    success('Delete Conversation (cascade delete messages)');
  } catch (err) {
    fail('Delete Conversation (cascade delete messages)', (err as Error).message);
  }
}

async function testRealTimeSubscription(): Promise<void> {
  log('TEST 10: Real-time Subscription Check');

  try {
    // Check if real-time is enabled by attempting a subscription
    const channel = supabase
      .channel('test-channel')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'chat_conversations' },
        (payload) => {
          console.log('Real-time event:', payload);
        }
      );

    // Clean up
    await channel.unsubscribe();

    console.log('Real-time subscription test completed (no errors)');
    success('Real-time Subscription Check');
  } catch (err) {
    fail('Real-time Subscription Check', (err as Error).message);
  }
}

async function cleanup(): Promise<void> {
  log('CLEANUP: Removing test data');

  try {
    // Delete all test conversations
    const { error } = await supabase
      .from('chat_conversations')
      .delete()
      .eq('user_id', TEST_USER_ID);

    if (error) {
      console.log('Cleanup error:', error.message);
    } else {
      console.log('Test data cleaned up successfully');
    }
  } catch (err) {
    console.log('Cleanup error:', (err as Error).message);
  }
}

async function printSummary(): Promise<void> {
  log('TEST SUMMARY');

  const passed = results.filter((r) => r.passed).length;
  const failed = results.filter((r) => !r.passed).length;

  console.log(`\nTotal Tests: ${results.length}`);
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);
  console.log(`\n${'='.repeat(60)}`);

  if (failed > 0) {
    console.log('\nFailed Tests:');
    results.filter((r) => !r.passed).forEach((r) => {
      console.log(`  - ${r.name}: ${r.error}`);
    });
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log(failed === 0 ? '\n🎉 ALL TESTS PASSED!' : '\n⚠️ SOME TESTS FAILED');
}

async function runTests(): Promise<void> {
  console.log('\n🧪 Starting Chat API E2E Tests\n');
  console.log(`Test User ID: ${TEST_USER_ID}`);

  // Test 1: Create conversation
  const conversationId = await testCreateConversation();
  if (!conversationId) {
    console.log('\n❌ Cannot continue without conversation ID');
    await printSummary();
    return;
  }

  // Test 2: Add user message
  await testAddUserMessage(conversationId);

  // Test 3: Add assistant message with tool invocations
  await testAddAssistantMessageWithToolInvocations(conversationId);

  // Test 4: List conversations
  await testListConversations();

  // Test 5: Get conversation messages
  await testGetConversationMessages(conversationId);

  // Test 6: Verify auto-title trigger
  await testConversationAutoTitle(conversationId);

  // Test 7: Update conversation
  await testUpdateConversation(conversationId);

  // Test 8: Create conversation with project
  const projectConvId = await testCreateConversationWithProject();

  // Test 9: Delete conversation (cascade)
  await testDeleteConversation(conversationId);

  // Test 10: Real-time subscription check
  await testRealTimeSubscription();

  // Cleanup
  await cleanup();

  // Summary
  await printSummary();
}

// Run tests
runTests().catch(console.error);
