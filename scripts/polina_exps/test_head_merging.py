#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работоспособности объединения attention-голов.
Использует упрощенную версию с меньшим количеством данных для быстрого тестирования.
"""

import torch
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from merge_attention_heads import AttentionHeadMerger
from pruninghealing.dataset import DatasetLoader

def test_basic_functionality():
    """Тест базовой функциональности"""
    print("Testing basic functionality...")
    
    # Проверяем доступность модели
    model_path = "src/checkpoints/llama3.1-8b"
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} not found. Using alternative model for testing.")
        model_path = "microsoft/DialoGPT-small"  # Маленькая модель для тестирования
    
    try:
        # Инициализируем merger
        merger = AttentionHeadMerger(model_path, device="cpu")  # Используем CPU для тестирования
        print("✓ Model loaded successfully")
        
        # Тестируем извлечение весов
        weights = merger.extract_attention_weights(0)
        print(f"✓ Extracted weights for layer 0: {weights['num_heads']} heads")
        
        # Тестируем вычисление схожести
        similarity = merger.compute_head_similarity(0)
        print(f"✓ Computed similarity matrix: {similarity.shape}")
        
        # Тестируем поиск пар
        pairs = merger.find_similar_pairs(similarity, threshold=0.5)  # Низкий порог для тестирования
        print(f"✓ Found {len(pairs)} similar pairs with threshold 0.5")
        
        # Тестируем объединение (если есть пары)
        if pairs:
            merger.merge_heads(0, pairs[:1])  # Объединяем только одну пару
            print("✓ Head merging completed")
            
            # Восстанавливаем веса
            merger.restore_weights(0)
            print("✓ Weights restored")
        
        print("✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

def test_dataset_loading():
    """Тест загрузки датасета"""
    print("\nTesting dataset loading...")
    
    try:
        from transformers import AutoTokenizer
        
        # Используем простой токенизатор для тестирования
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Тестируем загрузку датасета
        dataset_loader = DatasetLoader(tokenizer)
        dataset_loader.load_wikitext(max_length=128, train_size=10, eval_size=5)
        
        print(f"✓ Dataset loaded: {len(dataset_loader.train_dataset)} train, {len(dataset_loader.eval_dataset)} eval")
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {str(e)}")
        return False

def test_similarity_computation():
    """Тест вычисления схожести"""
    print("\nTesting similarity computation...")
    
    try:
        # Создаем тестовые данные
        np.random.seed(42)
        num_heads = 8
        head_dim = 64
        
        # Создаем случайные веса голов
        head_weights = np.random.randn(num_heads, head_dim * 128)
        
        # Делаем некоторые головы похожими
        head_weights[1] = head_weights[0] + 0.01 * np.random.randn(head_dim * 128)
        head_weights[3] = head_weights[2] + 0.01 * np.random.randn(head_dim * 128)
        
        # Вычисляем схожесть
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(head_weights)
        
        # Проверяем результаты
        assert similarity.shape == (num_heads, num_heads)
        assert np.allclose(np.diag(similarity), 1.0)  # Диагональ должна быть 1
        assert similarity[0, 1] > 0.9  # Похожие головы должны иметь высокую схожесть
        
        print(f"✓ Similarity computation works correctly")
        print(f"  - Matrix shape: {similarity.shape}")
        print(f"  - Max off-diagonal similarity: {np.max(similarity - np.eye(num_heads)):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Similarity test failed: {str(e)}")
        return False

def main():
    """Запуск всех тестов"""
    print("=" * 50)
    print("HEAD MERGING FUNCTIONALITY TESTS")
    print("=" * 50)
    
    tests = [
        test_similarity_computation,
        test_dataset_loading,
        test_basic_functionality,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The system is ready for experiments.")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    exit(main())