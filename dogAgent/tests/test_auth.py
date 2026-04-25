"""
dogAgent 认证模块测试

测试内容：
1. 用户注册（正常 + 重复 + 参数校验）
2. 用户登录（正确密码 + 错误密码）
3. JWT 签发与验证
4. 多用户数据隔离
"""

import sys
import os

# 确保项目根目录在 path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from agent.memory import MemoryDB, ConversationStore
from agent.session import SessionManager
from api.auth import AuthService


def test_register():
    """测试用户注册"""
    db = MemoryDB(":memory:")
    auth = AuthService(db)

    # 正常注册
    user = auth.register("alice", "pass1234", "Alice Wang")
    assert user["username"] == "alice"
    assert user["display_name"] == "Alice Wang"
    assert len(user["id"]) > 0
    print("  ✅ 正常注册成功")

    # 重复注册
    try:
        auth.register("alice", "other_pass")
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "已被注册" in str(e)
    print("  ✅ 重复注册被拒绝")

    # 用户名太短
    try:
        auth.register("a", "pass1234")
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass
    print("  ✅ 短用户名被拒绝")

    # 密码太短
    try:
        auth.register("bob", "123")
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass
    print("  ✅ 短密码被拒绝")

    # 不传 display_name 时默认用 username
    user2 = auth.register("bob", "pass5678")
    assert user2["display_name"] == "bob"
    print("  ✅ 默认 display_name 正确")

    # 用户名大小写不敏感
    try:
        auth.register("Alice", "pass9999")
        assert False, "大小写不同的用户名应被视为重复"
    except ValueError:
        pass
    print("  ✅ 用户名大小写不敏感")

    db.close()
    print("✅ test_register 全部通过")


def test_authenticate():
    """测试用户登录"""
    db = MemoryDB(":memory:")
    auth = AuthService(db)

    auth.register("charlie", "secret123", "Charlie")

    # 正确密码
    user = auth.authenticate("charlie", "secret123")
    assert user is not None
    assert user["username"] == "charlie"
    assert user["display_name"] == "Charlie"
    print("  ✅ 正确密码登录成功")

    # 错误密码
    result = auth.authenticate("charlie", "wrong_password")
    assert result is None
    print("  ✅ 错误密码被拒绝")

    # 不存在的用户
    result = auth.authenticate("nonexistent", "pass123")
    assert result is None
    print("  ✅ 不存在用户被拒绝")

    # 大小写不敏感
    user2 = auth.authenticate("Charlie", "secret123")
    assert user2 is not None
    print("  ✅ 用户名大小写不敏感登录")

    db.close()
    print("✅ test_authenticate 全部通过")


def test_jwt():
    """测试 JWT 签发与验证"""
    db = MemoryDB(":memory:")
    auth = AuthService(db)

    user = auth.register("dave", "pass1234")

    # 签发
    token = auth.create_token(user["id"], user["username"])
    assert isinstance(token, str)
    assert len(token) > 20
    print(f"  ✅ JWT 签发成功: {token[:40]}...")

    # 验证
    payload = auth.verify_token(token)
    assert payload is not None
    assert payload["user_id"] == user["id"]
    assert payload["username"] == "dave"
    print("  ✅ JWT 验证成功")

    # 无效 token
    result = auth.verify_token("invalid.token.string")
    assert result is None
    print("  ✅ 无效 JWT 被拒绝")

    # 篡改 token
    tampered = token[:-5] + "XXXXX"
    result = auth.verify_token(tampered)
    assert result is None
    print("  ✅ 篡改 JWT 被拒绝")

    # 空 token
    result = auth.verify_token("")
    assert result is None
    print("  ✅ 空 JWT 被拒绝")

    db.close()
    print("✅ test_jwt 全部通过")


def test_get_user_by_id():
    """测试根据 ID 获取用户"""
    db = MemoryDB(":memory:")
    auth = AuthService(db)

    user = auth.register("eve", "pass1234", "Eve")

    # 正常获取
    found = auth.get_user_by_id(user["id"])
    assert found is not None
    assert found["username"] == "eve"
    assert found["display_name"] == "Eve"
    print("  ✅ 根据 ID 获取用户成功")

    # 不存在的 ID
    not_found = auth.get_user_by_id("nonexistent-id-12345")
    assert not_found is None
    print("  ✅ 不存在 ID 返回 None")

    db.close()
    print("✅ test_get_user_by_id 全部通过")


def test_multi_user_isolation():
    """测试多用户数据隔离"""
    db = MemoryDB(":memory:")
    auth = AuthService(db)
    conv = ConversationStore(db)
    sm = SessionManager(db)

    # 注册两个用户
    user1 = auth.register("user_a", "pass1111", "用户A")
    user2 = auth.register("user_b", "pass2222", "用户B")
    print("  ✅ 两个用户注册成功")

    # 各自创建会话
    sid1 = sm.create_session(user1["id"])
    sid2 = sm.create_session(user2["id"])
    print(f"  ✅ 会话创建: A={sid1[:8]}..., B={sid2[:8]}...")

    # 各自添加消息
    conv.add_message(user1["id"], sid1, "user", "我的狗叫旺财，是迷你雪纳瑞")
    conv.add_message(user1["id"], sid1, "assistant", "旺财是个可爱的名字！")
    conv.add_message(user2["id"], sid2, "user", "我的狗叫小白，是标准雪纳瑞")
    conv.add_message(user2["id"], sid2, "assistant", "小白是个好名字！")

    # 用户A只能看到自己的消息
    msgs_a = conv.get_recent(user1["id"], sid1)
    assert len(msgs_a) == 2
    assert "旺财" in msgs_a[0]["content"]
    print("  ✅ 用户A看到自己的消息")

    # 用户B只能看到自己的消息
    msgs_b = conv.get_recent(user2["id"], sid2)
    assert len(msgs_b) == 2
    assert "小白" in msgs_b[0]["content"]
    print("  ✅ 用户B看到自己的消息")

    # 用户A看不到用户B的会话消息
    cross = conv.get_recent(user1["id"], sid2)
    assert len(cross) == 0
    print("  ✅ 用户A看不到用户B的消息")

    # 用户B看不到用户A的会话消息
    cross2 = conv.get_recent(user2["id"], sid1)
    assert len(cross2) == 0
    print("  ✅ 用户B看不到用户A的消息")

    # 各自的会话列表互不干扰
    sessions_a = sm.list_sessions(user1["id"])
    sessions_b = sm.list_sessions(user2["id"])
    assert len(sessions_a) == 1
    assert len(sessions_b) == 1
    assert sessions_a[0]["id"] != sessions_b[0]["id"]
    print("  ✅ 会话列表互不干扰")

    db.close()
    print("✅ test_multi_user_isolation 全部通过")


def test_full_auth_flow():
    """测试完整认证流程: 注册 → 签发token → 验证token → 获取用户"""
    db = MemoryDB(":memory:")
    auth = AuthService(db)

    # 1. 注册
    user = auth.register("frank", "mypassword", "Frank Li")
    print(f"  ✅ 注册: {user['username']}")

    # 2. 登录获取信息
    logged = auth.authenticate("frank", "mypassword")
    assert logged is not None
    print(f"  ✅ 登录: {logged['display_name']}")

    # 3. 签发 JWT
    token = auth.create_token(logged["id"], logged["username"])
    print(f"  ✅ 签发 token")

    # 4. 验证 JWT
    payload = auth.verify_token(token)
    assert payload["user_id"] == logged["id"]
    print(f"  ✅ 验证 token")

    # 5. 通过 user_id 获取完整用户信息
    full = auth.get_user_by_id(payload["user_id"])
    assert full["display_name"] == "Frank Li"
    print(f"  ✅ 获取用户: {full['display_name']}")

    db.close()
    print("✅ test_full_auth_flow 全部通过")


if __name__ == "__main__":
    print("=" * 50)
    print("dogAgent 认证模块测试")
    print("=" * 50)
    print()

    tests = [
        ("注册功能", test_register),
        ("登录功能", test_authenticate),
        ("JWT 签发验证", test_jwt),
        ("ID 查询用户", test_get_user_by_id),
        ("多用户数据隔离", test_multi_user_isolation),
        ("完整认证流程", test_full_auth_flow),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        print(f"\n--- {name} ---")
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 50)
    total = passed + failed
    print(f"结果: {passed}/{total} 通过, {failed} 失败")
    if failed == 0:
        print("🎉 全部测试通过！")
    else:
        print("⚠️ 有测试失败，请检查")
        sys.exit(1)