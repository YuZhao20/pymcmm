# GitHub Push Instructions for v0.2.0

## 前提条件

1. GitHubリポジトリが既に設定されていること
2. リモートリポジトリが `origin` として設定されていること
3. 現在のブランチが `main` であること

## 手順

### 1. 変更をステージング

```bash
git add -A
```

### 2. コミット

```bash
git commit -m "v0.2.0: Add Cython acceleration (35x speedup)

- Add Cython implementation for core math functions
- Automatic fallback to pure Python when Cython unavailable
- Add check_acceleration() function
- Improve numerical stability
- Add test suite"
```

### 3. タグの作成

```bash
git tag -a v0.2.0 -m "Version 0.2.0 - Cython acceleration"
```

### 4. プッシュ

```bash
# メインブランチをプッシュ
git push origin main

# タグをプッシュ
git push origin v0.2.0
```

### 5. GitHub Releaseの作成

1. GitHubのリポジトリページにアクセス
2. "Releases" セクションをクリック
3. "Draft a new release" をクリック
4. 以下の情報を入力：
   - **Tag**: `v0.2.0` を選択
   - **Title**: `v0.2.0 - Cython Acceleration`
   - **Description**: `CHANGELOG.md` の内容をコピー＆ペースト
5. "Publish release" をクリック

## 確認

- コミットが正常にプッシュされたか確認
- タグが正常にプッシュされたか確認
- Releaseが正常に作成されたか確認

