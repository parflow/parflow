# macOS release packaging (Path B)

Signed distribution uses a minimal **ParFlow.app** Xcode wrapper around the
CMake `install/` prefix produced by CI.

## Layout

- `ParFlow.xcodeproj` — macOS app target (`org.parflow.ParFlow`)
- `ParFlow/main.c` — launcher executable (`Contents/MacOS/ParFlow`)
- `ParFlow/Resources/parflow/` — populated at CI time (not committed)
- `ExportOptions.plist` — `developer-id` + automatic signing
- `stage-app-payload.sh` — copy `$PARFLOW_DIR` into Resources

## Local test (macOS with API key)

```bash
# After a normal install to ~/install:
./packaging/macos/stage-app-payload.sh ~/install

export APP_STORE_CONNECT_KEY_PATH=/path/to/AuthKey.p8
xcodebuild archive \
  -project packaging/macos/ParFlow.xcodeproj \
  -scheme ParFlow -configuration Release \
  -archivePath build/ParFlow.xcarchive \
  -allowProvisioningUpdates \
  -authenticationKeyPath "$APP_STORE_CONNECT_KEY_PATH" \
  -authenticationKeyID "<KEY_ID>" \
  -authenticationKeyIssuerID "<ISSUER_ID>"

xcodebuild -exportArchive \
  -archivePath build/ParFlow.xcarchive \
  -exportOptionsPlist packaging/macos/ExportOptions.plist \
  -exportPath build/export \
  -authenticationKeyPath "$APP_STORE_CONNECT_KEY_PATH" \
  -authenticationKeyID "<KEY_ID>" \
  -authenticationKeyIssuerID "<ISSUER_ID>"
```

## Team ID

`Y3TW367T4G` is set in `ExportOptions.plist` and the Xcode project. Update both
if the distribution team changes.
