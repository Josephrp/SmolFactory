#!/usr/bin/env python3
"""
Comprehensive TRL compatibility test
Verifies all TRL interface requirements are met
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_core_interface():
    """Test core TRL interface requirements"""
    print("🧪 Testing Core TRL Interface...")
    
    try:
        import trackio
        
        # Test 1: Core functions exist
        required_functions = ['init', 'log', 'finish']
        for func_name in required_functions:
            assert hasattr(trackio, func_name), f"trackio.{func_name} not found"
            print(f"✅ trackio.{func_name} exists")
        
        # Test 2: Config attribute exists
        assert hasattr(trackio, 'config'), "trackio.config not found"
        print("✅ trackio.config exists")
        
        # Test 3: Config has update method
        config = trackio.config
        assert hasattr(config, 'update'), "trackio.config.update not found"
        print("✅ trackio.config.update exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Core interface test failed: {e}")
        return False

def test_init_functionality():
    """Test init function with various argument patterns"""
    print("\n🔧 Testing Init Functionality...")
    
    try:
        import trackio
        
        # Test 1: No arguments (TRL compatibility)
        try:
            experiment_id = trackio.init()
            print(f"✅ trackio.init() without args: {experiment_id}")
        except Exception as e:
            print(f"❌ trackio.init() without args failed: {e}")
            return False
        
        # Test 2: With arguments
        try:
            experiment_id = trackio.init(project_name="test_project", experiment_name="test_exp")
            print(f"✅ trackio.init() with args: {experiment_id}")
        except Exception as e:
            print(f"❌ trackio.init() with args failed: {e}")
            return False
        
        # Test 3: With kwargs
        try:
            experiment_id = trackio.init(test_param="test_value")
            print(f"✅ trackio.init() with kwargs: {experiment_id}")
        except Exception as e:
            print(f"❌ trackio.init() with kwargs failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Init functionality test failed: {e}")
        return False

def test_log_functionality():
    """Test log function with various metric types"""
    print("\n📊 Testing Log Functionality...")
    
    try:
        import trackio
        
        # Test 1: Basic metrics
        try:
            trackio.log({'loss': 0.5, 'accuracy': 0.8})
            print("✅ trackio.log() with basic metrics")
        except Exception as e:
            print(f"❌ trackio.log() with basic metrics failed: {e}")
            return False
        
        # Test 2: With step parameter
        try:
            trackio.log({'loss': 0.4, 'lr': 1e-4}, step=100)
            print("✅ trackio.log() with step parameter")
        except Exception as e:
            print(f"❌ trackio.log() with step failed: {e}")
            return False
        
        # Test 3: TRL-specific metrics
        try:
            trackio.log({
                'total_tokens': 1000,
                'truncated_tokens': 50,
                'padding_tokens': 20,
                'throughput': 100.5,
                'step_time': 0.1
            })
            print("✅ trackio.log() with TRL-specific metrics")
        except Exception as e:
            print(f"❌ trackio.log() with TRL metrics failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Log functionality test failed: {e}")
        return False

def test_config_update():
    """Test config update with TRL-specific patterns"""
    print("\n⚙️ Testing Config Update...")
    
    try:
        import trackio
        
        config = trackio.config
        
        # Test 1: TRL-specific keyword arguments
        try:
            config.update(allow_val_change=True, project_name="trl_test")
            print(f"✅ Config update with TRL kwargs: allow_val_change={config.allow_val_change}")
        except Exception as e:
            print(f"❌ Config update with TRL kwargs failed: {e}")
            return False
        
        # Test 2: Dictionary update
        try:
            config.update({'experiment_name': 'test_exp', 'new_param': 'value'})
            print(f"✅ Config update with dict: experiment_name={config.experiment_name}")
        except Exception as e:
            print(f"❌ Config update with dict failed: {e}")
            return False
        
        # Test 3: Mixed update
        try:
            config.update({'mixed_param': 'dict_value'}, kwarg_param='keyword_value')
            print(f"✅ Config update with mixed args: mixed_param={config.mixed_param}, kwarg_param={config.kwarg_param}")
        except Exception as e:
            print(f"❌ Config update with mixed args failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config update test failed: {e}")
        return False

def test_finish_functionality():
    """Test finish function"""
    print("\n🏁 Testing Finish Functionality...")
    
    try:
        import trackio
        
        # Test finish function
        try:
            trackio.finish()
            print("✅ trackio.finish() completed successfully")
        except Exception as e:
            print(f"❌ trackio.finish() failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Finish functionality test failed: {e}")
        return False

def test_trl_trainer_simulation():
    """Simulate TRL trainer usage patterns"""
    print("\n🤖 Testing TRL Trainer Simulation...")
    
    try:
        import trackio
        
        # Simulate SFTTrainer initialization
        try:
            # Initialize trackio (like TRL does)
            experiment_id = trackio.init()
            print(f"✅ TRL-style initialization: {experiment_id}")
            
            # Update config (like TRL does)
            trackio.config.update(allow_val_change=True, project_name="trl_simulation")
            print("✅ TRL-style config update")
            
            # Log metrics (like TRL does during training)
            for step in range(1, 4):
                trackio.log({
                    'loss': 1.0 / step,
                    'learning_rate': 1e-4,
                    'total_tokens': step * 1000,
                    'throughput': 100.0 / step
                }, step=step)
                print(f"✅ TRL-style logging at step {step}")
            
            # Finish experiment (like TRL does)
            trackio.finish()
            print("✅ TRL-style finish")
            
        except Exception as e:
            print(f"❌ TRL trainer simulation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ TRL trainer simulation test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and fallbacks"""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        import trackio
        
        # Test 1: Graceful handling of missing monitor
        try:
            # This should not crash even if monitor is not available
            trackio.log({'test': 1.0})
            print("✅ Graceful handling of logging without monitor")
        except Exception as e:
            print(f"⚠️ Logging without monitor: {e}")
            # This is acceptable - just a warning
        
        # Test 2: Config update with invalid data
        try:
            config = trackio.config
            config.update(invalid_param=None)
            print("✅ Config update with invalid data handled gracefully")
        except Exception as e:
            print(f"❌ Config update with invalid data failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_dict_style_access():
    """Test dictionary-style access to TrackioConfig"""
    print("\n📝 Testing Dictionary-Style Access...")
    
    try:
        import trackio
        
        config = trackio.config
        
        # Test 1: Dictionary-style assignment
        try:
            config['test_key'] = 'test_value'
            print(f"✅ Dictionary assignment: test_key={config['test_key']}")
        except Exception as e:
            print(f"❌ Dictionary assignment failed: {e}")
            return False
        
        # Test 2: Dictionary-style access
        try:
            value = config['test_key']
            print(f"✅ Dictionary access: {value}")
        except Exception as e:
            print(f"❌ Dictionary access failed: {e}")
            return False
        
        # Test 3: Contains check
        try:
            has_key = 'test_key' in config
            print(f"✅ Contains check: {'test_key' in config}")
        except Exception as e:
            print(f"❌ Contains check failed: {e}")
            return False
        
        # Test 4: Get method
        try:
            value = config.get('test_key', 'default')
            default_value = config.get('nonexistent', 'default')
            print(f"✅ Get method: {value}, default: {default_value}")
        except Exception as e:
            print(f"❌ Get method failed: {e}")
            return False
        
        # Test 5: TRL-style usage
        try:
            config['allow_val_change'] = True
            config['report_to'] = 'trackio'
            print(f"✅ TRL-style config: allow_val_change={config['allow_val_change']}, report_to={config['report_to']}")
        except Exception as e:
            print(f"❌ TRL-style config failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dictionary-style access test failed: {e}")
        return False

def main():
    """Run comprehensive TRL compatibility tests"""
    print("🧪 Comprehensive TRL Compatibility Test")
    print("=" * 50)
    
    tests = [
        ("Core Interface", test_core_interface),
        ("Init Functionality", test_init_functionality),
        ("Log Functionality", test_log_functionality),
        ("Config Update", test_config_update),
        ("Finish Functionality", test_finish_functionality),
        ("TRL Trainer Simulation", test_trl_trainer_simulation),
        ("Error Handling", test_error_handling),
        ("Dictionary-Style Access", test_dict_style_access),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TRL Compatibility Test Results")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! TRL compatibility is complete.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 