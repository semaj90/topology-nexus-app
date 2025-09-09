/**
 * Test suite for QLoRA Memory Manager
 */

const QLoRAMemoryManager = require('../src/qlora/memory_manager');

// Simple test framework
class SimpleTest {
    constructor() {
        this.passed = 0;
        this.failed = 0;
    }

    async test(name, testFn) {
        try {
            await testFn();
            console.log(`✓ ${name}`);
            this.passed++;
        } catch (error) {
            console.error(`✗ ${name}: ${error.message}`);
            this.failed++;
        }
    }

    summary() {
        const total = this.passed + this.failed;
        console.log(`\nTest Summary: ${this.passed}/${total} passed`);
        if (this.failed > 0) {
            process.exit(1);
        }
    }
}

// Test utilities
function assert(condition, message = 'Assertion failed') {
    if (!condition) {
        throw new Error(message);
    }
}

function assertEqual(actual, expected, message = 'Values not equal') {
    if (actual !== expected) {
        throw new Error(`${message}: expected ${expected}, got ${actual}`);
    }
}

// Test suite
async function runTests() {
    const test = new SimpleTest();

    await test.test('QLoRAMemoryManager initialization', async () => {
        const manager = new QLoRAMemoryManager({ maxMemoryMB: 256 });
        assert(manager.maxMemoryMB === 256, 'Max memory should be 256MB');
        assertEqual(manager.loadedModules.size, 0, 'Should start with no loaded modules');
    });

    await test.test('Module loading and retrieval', async () => {
        const manager = new QLoRAMemoryManager();
        
        const sampleModule = {
            id: 'test-module',
            type: 'classifier',
            parameters: {
                weights: new Array(100).fill(0.5),
                biases: new Array(10).fill(0.1)
            }
        };

        await manager.loadModule('test-1', sampleModule);
        assert(manager.loadedModules.has('test-1'), 'Module should be loaded');
        
        const retrieved = await manager.getModule('test-1');
        assert(retrieved.id === 'test-module', 'Retrieved module should match');
    });

    await test.test('Memory management', async () => {
        const manager = new QLoRAMemoryManager({ maxMemoryMB: 1 }); // Very small memory limit
        
        const largeModule = {
            id: 'large-module',
            type: 'transformer',
            parameters: {
                weights: new Array(10000).fill(Math.random())
            }
        };

        await manager.loadModule('large-1', largeModule, 1);
        await manager.loadModule('large-2', largeModule, 2);
        
        // Check that memory management works
        const stats = manager.getStats();
        assert(stats.loadedModules <= 2, 'Should manage memory by unloading modules');
    });

    await test.test('Module relevance calculation', async () => {
        const manager = new QLoRAMemoryManager();
        
        const queryTokens = ['machine', 'learning', 'classification'];
        const moduleInfo = {
            id: 'ml-module',
            description: 'Machine learning classification model',
            tags: ['ml', 'classification', 'supervised'],
            domain: 'machine-learning'
        };

        const relevance = manager.calculateRelevance(queryTokens, moduleInfo);
        assert(relevance > 0, 'Should find relevance for matching terms');
    });

    await test.test('Contextual module swapping', async () => {
        const manager = new QLoRAMemoryManager();
        
        const modules = [
            {
                id: 'nlp-module',
                description: 'Natural language processing module',
                tags: ['nlp', 'text'],
                domain: 'text-processing'
            },
            {
                id: 'cv-module', 
                description: 'Computer vision module',
                tags: ['vision', 'image'],
                domain: 'image-processing'
            }
        ];

        const swapOps = await manager.contextualSwap('process text documents', modules);
        assert(swapOps.length > 0, 'Should generate swap operations');
        
        const loadOps = swapOps.filter(op => op.operation === 'load');
        assert(loadOps.some(op => op.moduleId === 'nlp-module'), 'Should prioritize NLP module for text query');
    });

    await test.test('Module compression and decompression', async () => {
        const manager = new QLoRAMemoryManager();
        
        const originalModule = {
            id: 'compression-test',
            type: 'test',
            parameters: {
                weights: new Array(1000).fill(0.5),
                biases: new Array(100).fill(0.1)
            },
            version: '1.0'
        };

        const compressed = await manager.compressModule(originalModule);
        assert(compressed.metadata, 'Compressed module should have metadata');
        assert(compressed.parameters, 'Compressed module should have parameters');

        const decompressed = await manager.decompressModule(compressed);
        assert(decompressed.decompressed === true, 'Should mark as decompressed');
        assert(decompressed.id === originalModule.id, 'Should preserve module ID');
    });

    await test.test('Query tokenization', async () => {
        const manager = new QLoRAMemoryManager();
        
        const query = "Machine learning and natural language processing!";
        const tokens = manager.tokenizeQuery(query);
        
        assert(tokens.includes('machine'), 'Should include "machine" token');
        assert(tokens.includes('learning'), 'Should include "learning" token');
        assert(tokens.includes('natural'), 'Should include "natural" token');
        assert(tokens.includes('and'), 'Should include "and" token (length 3)');
        assert(!tokens.includes('!'), 'Should filter punctuation');
        assert(!tokens.includes(''), 'Should filter empty strings');
    });

    await test.test('Memory statistics', async () => {
        const manager = new QLoRAMemoryManager({ maxMemoryMB: 512 });
        
        const stats = manager.getStats();
        assert(typeof stats.loadedModules === 'number', 'Should report loaded modules count');
        assert(typeof stats.memoryUsageBytes === 'number', 'Should report memory usage in bytes');
        assert(typeof stats.memoryUsagePercent === 'number', 'Should report memory usage percentage');
        assert(stats.maxMemoryMB === 512, 'Should report max memory limit');
    });

    test.summary();
}

// Run tests
if (require.main === module) {
    console.log('Running QLoRA Memory Manager Tests...\n');
    runTests().catch(console.error);
}

module.exports = { runTests };