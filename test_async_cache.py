#!/usr/bin/env python3
"""
测试异步处理和缓存功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import numpy as np
import time
from api.cache_manager import frame_cache, detection_cache, vlm_result_cache, compute_frame_hash
from api.event_deduplicator import event_deduplicator
from api.async_processor import AsyncVideoProcessor

async def test_cache_basic():
    """测试缓存基本功能"""
    print('=' * 50)
    print('测试1: 缓存基本功能')
    print('=' * 50)
    
    # 测试帧哈希计算
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame_hash = compute_frame_hash(frame)
    print(f'帧哈希计算: {frame_hash[:16]}... (完整长度: {len(frame_hash)})')
    
    # 测试缓存设置和获取
    frame_cache.set_detection(frame_hash, {'boxes': [[100, 100, 200, 200]], 'confidence': 0.9})
    cached = frame_cache.get_detection(frame_hash)
    assert cached is not None, "缓存获取失败"
    print('缓存设置和获取: ✅')
    
    # 测试缓存命中率
    for i in range(10):
        result = frame_cache.get_detection(frame_hash)
    
    stats = frame_cache.get_stats()
    print(f'缓存统计: {stats}')
    assert stats['detection']['hits'] >= 9, "命中率不足"
    print('命中率测试: ✅')

async def test_cache_ttl():
    """测试缓存过期时间"""
    print('\n' + '=' * 50)
    print('测试2: 缓存TTL过期')
    print('=' * 50)
    
    # 设置一个短TTL的缓存
    test_key = 'test_ttl_key'
    detection_cache.set('ttl_test', 'test_value', test_key, ttl=1)
    
    # 立即获取应该存在
    result = detection_cache.get('ttl_test', test_key)
    assert result == 'test_value', "缓存未设置成功"
    print('设置缓存: ✅')
    
    # 等待2秒后应该过期
    await asyncio.sleep(2)
    result = detection_cache.get('ttl_test', test_key)
    assert result is None, "缓存未过期"
    print('TTL过期测试: ✅')

async def test_cache_eviction():
    """测试缓存淘汰策略"""
    print('\n' + '=' * 50)
    print('测试3: 缓存淘汰策略')
    print('=' * 50)
    
    # 填满缓存
    for i in range(10):
        detection_cache.set('eviction_test', f'value_{i}', f'key_{i}', ttl=300)
    
    stats = detection_cache.get_stats()
    print(f'缓存填充后: {stats}')
    
    # 添加更多数据触发淘汰
    for i in range(10, 20):
        detection_cache.set('eviction_test', f'value_{i}', f'key_{i}', ttl=300)
    
    stats = detection_cache.get_stats()
    print(f'触发淘汰后: {stats}')
    assert stats['size'] <= detection_cache.max_size, "缓存超过最大限制"
    print('缓存淘汰测试: ✅')

async def test_deduplication():
    """测试智能去重"""
    print('\n' + '=' * 50)
    print('测试4: 智能事件去重')
    print('=' * 50)
    
    # 同一位置重复事件应该被过滤
    result1 = event_deduplicator.record_event('fall', 1, bbox=(100, 100, 200, 200), confidence=0.9)
    result2 = event_deduplicator.record_event('fall', 1, bbox=(105, 105, 205, 205), confidence=0.85)
    
    assert result1 == True, "第一个事件应该记录成功"
    assert result2 == False, "相近位置的重复事件应该被过滤"
    print('空间去重测试: ✅')
    
    # 不同位置事件应该通过
    result3 = event_deduplicator.record_event('fall', 1, bbox=(500, 500, 600, 600), confidence=0.9)
    assert result3 == True, "不同位置事件应该记录成功"
    print('不同位置测试: ✅')
    
    # 频率衰减测试
    for i in range(5):
        result = event_deduplicator.record_event('loitering', 1, bbox=(300, 300, 400, 400), confidence=0.6)
        if i == 0:
            assert result == True, "第一次徘徊事件应该记录"
        else:
            assert result == False, f"第{i+1}次徘徊事件应该被过滤"
    
    stats = event_deduplicator.get_stats()
    print(f'去重器统计: {stats}')
    print('频率衰减测试: ✅')

async def test_async_processor():
    """测试异步视频处理器"""
    print('\n' + '=' * 50)
    print('测试5: 异步视频处理器')
    print('=' * 50)
    
    processor = AsyncVideoProcessor(max_workers=2, use_process_pool=False)
    
    # 测试启动和停止
    await processor.start()
    print('异步处理器启动: ✅')
    
    await processor.stop()
    print('异步处理器停止: ✅')

async def test_vlm_cache():
    """测试VLM缓存"""
    print('\n' + '=' * 50)
    print('测试6: VLM结果缓存')
    print('=' * 50)
    
    video_hash = 'test_video_hash_001'
    
    # 设置VLM缓存
    vlm_result_cache.set('describe', '这是一个视频场景描述', video_hash)
    
    # 获取缓存
    result = vlm_result_cache.get('describe', video_hash)
    assert result == '这是一个视频场景描述', "VLM缓存获取失败"
    print('VLM缓存设置和获取: ✅')
    
    # 测试缓存统计
    stats = vlm_result_cache.get_stats()
    print(f'VLM缓存统计: {stats}')

async def main():
    """主测试函数"""
    print('\n' + '=' * 60)
    print('智能视频分析系统 - 异步处理和缓存功能测试')
    print('=' * 60)
    
    try:
        await test_cache_basic()
        await test_cache_ttl()
        await test_cache_eviction()
        await test_deduplication()
        await test_async_processor()
        await test_vlm_cache()
        
        print('\n' + '=' * 60)
        print('🎉 所有测试通过!')
        print('=' * 60)
        
    except AssertionError as e:
        print(f'\n❌ 测试失败: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ 测试异常: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())