// App.js — Expo версия "garbage classifier"
// Камера/галерея → превью → "Classify the item/items" → лоадер → показ картинки-ответа или Retry.
// Байты готовятся как Uint8Array. Есть 2 закомментированных варианта интеграции с FastAPI.

import React, { useCallback, useMemo, useState } from 'react';
import { ActivityIndicator, Alert, Image, Pressable, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import * as Base64 from 'base64-js'; // для надёжной конвертации bytes<->base64

// Изумрудная минималистичная тема
const COLORS = {
  bg: '#0B0F0E',
  fg: '#ECFDF5',
  emerald: '#10B981',
  emeraldDark: '#059669',
  gray: '#9CA3AF',
  card: '#111827',
  danger: '#EF4444',
};

const PHASES = {
  CAPTURE: 'CAPTURE',
  PREVIEW: 'PREVIEW',
  PROCESSING: 'PROCESSING',
  RESULT: 'RESULT',
};

export default function App() {
  const [phase, setPhase] = useState(PHASES.CAPTURE);
  const [image, setImage] = useState(null);      // { uri, base64? }
  const [resultUri, setResultUri] = useState(null);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState(null);

  const canClassify = useMemo(() => !!image && phase === PHASES.PREVIEW, [image, phase]);

  // Разрешения
  const ensurePermissions = useCallback(async () => {
    const lib = await ImagePicker.requestMediaLibraryPermissionsAsync();
    const cam = await ImagePicker.requestCameraPermissionsAsync();
    const ok = lib.status === 'granted' && cam.status === 'granted';
    if (!ok) Alert.alert('Permissions', 'Разрешите доступ к камере и галерее.');
    return ok;
  }, []);

  // Камера
  const openCamera = useCallback(async () => {
    try {
      if (!(await ensurePermissions())) return;
      const res = await ImagePicker.launchCameraAsync({
        quality: 0.9,
        base64: true, // удобно: сразу получаем base64
      });
      if (res.canceled) return;
      const asset = res.assets?.[0];
      if (!asset?.uri) throw new Error('No camera image');
      setImage({ uri: asset.uri, base64: asset.base64 || null });
      setError(null);
      setPhase(PHASES.PREVIEW);
    } catch (e) {
      setError(e?.message || 'Camera error');
    }
  }, [ensurePermissions]);

  // Галерея
  const openGallery = useCallback(async () => {
    try {
      if (!(await ensurePermissions())) return;
      const res = await ImagePicker.launchImageLibraryAsync({
        quality: 0.95,
        base64: true,
        selectionLimit: 1,
      });
      if (res.canceled) return;
      const asset = res.assets?.[0];
      if (!asset?.uri) throw new Error('No gallery image');
      setImage({ uri: asset.uri, base64: asset.base64 || null });
      setError(null);
      setPhase(PHASES.PREVIEW);
    } catch (e) {
      setError(e?.message || 'Gallery error');
    }
  }, [ensurePermissions]);

  // utils
  const base64ToBytes = (b64) => Base64.toByteArray(b64); // Uint8Array
  const bytesToBase64 = (bytes) => Base64.fromByteArray(bytes); // string

  // Если picker не дал base64 — сконвертируем из uri
  const uriToBase64 = useCallback(async (uri) => {
    const resp = await fetch(uri);
    const buf = new Uint8Array(await resp.arrayBuffer());
    return bytesToBase64(buf);
  }, []);

  const resetAll = useCallback(() => {
    setPhase(PHASES.CAPTURE);
    setImage(null);
    setResultUri(null);
    setError(null);
  }, []);


  const classify = useCallback(async () => {
  if (!image?.uri) return;
    setPhase(PHASES.PROCESSING);
    setError(null);
    setResultUri(null);
    setDetections(null);

    try {
      // multipart/form-data → UploadFile = File(...) на FastAPI
      const form = new FormData();
      form.append('file', {
        uri: image.uri,
        name: 'photo.jpg',
        type: 'image/jpeg',
      });
      // return_image=true по умолчанию в нашем обработчике

      const resp = await fetch(API_URL, {
        method: 'POST',
        body: form,
        headers: { 'Accept': 'application/json' }, // ВАЖНО: не ставь Content-Type вручную
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();

      // json.annotated_image_b64 -> сохраняем в файл для <Image/>
      if (json.annotated_image_b64) {
        const outPath = FileSystem.cacheDirectory + `gc-result-${Date.now()}.jpg`;
        await FileSystem.writeAsStringAsync(outPath, json.annotated_image_b64, {
          encoding: FileSystem.EncodingType.Base64,
        });
        setResultUri(outPath);
      } else {
        // если картинка не пришла — показываем исходник, но это нетипично
        setResultUri(image.uri);
      }

      // детекции (массив объектов {bbox, yolo_conf, class, class_conf})
      setDetections(json.detections || []);
      setPhase(PHASES.RESULT);
    } catch (e) {
      setError(e?.message || 'Classification failed');
      setPhase(PHASES.RESULT);
    }
  }, [image]);

  // Основная логика "Classify"
  // const classify = useCallback(async () => {
  //   if (!image?.uri) return;
  //   setPhase(PHASES.PROCESSING);
  //   setError(null);
  //   setResultUri(null);

  //   try {
  //     // 1) Готовим байты
  //     let b64 = image.base64;
  //     if (!b64) b64 = await uriToBase64(image.uri);
  //     const bytes = base64ToBytes(b64); // Uint8Array — готово для FastAPI

  //     // ===================== ИНТЕГРАЦИЯ С FASTAPI =====================
  //     // Вариант A: application/octet-stream (сырая отправка байтов)
  //     // const resp = await fetch('http://<YOUR_FASTAPI_HOST>/classify', {
  //     //   method: 'POST',
  //     //   headers: { 'Content-Type': 'application/octet-stream' },
  //     //   body: bytes,
  //     // });
  //     // if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  //     // const outBytes = new Uint8Array(await resp.arrayBuffer());
  //     // const outB64 = bytesToBase64(outBytes);
  //     // const outPath = FileSystem.cacheDirectory + `gc-result-${Date.now()}.jpg`;
  //     // await FileSystem.writeAsStringAsync(outPath, outB64, { encoding: FileSystem.EncodingType.Base64 });
  //     // setResultUri(outPath);

  //     // Вариант B: multipart/form-data (UploadFile = File(...))
  //     // const form = new FormData();
  //     // form.append('file', {
  //     //   uri: image.uri,
  //     //   name: 'photo.jpg',
  //     //   type: 'image/jpeg',
  //     // });
  //     // const resp = await fetch('http://<YOUR_FASTAPI_HOST>/classify', { method: 'POST', body: form });
  //     // if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  //     // const outBytes = new Uint8Array(await resp.arrayBuffer());
  //     // const outB64 = bytesToBase64(outBytes);
  //     // const outPath = FileSystem.cacheDirectory + `gc-result-${Date.now()}.jpg`;
  //     // await FileSystem.writeAsStringAsync(outPath, outB64, { encoding: FileSystem.EncodingType.Base64 });
  //     // setResultUri(outPath);
  //     // =================== /ИНТЕГРАЦИЯ С FASTAPI ======================

  //     // ВРЕМЕННО: имитация ответа сервера (удали после интеграции)
  //     await new Promise((r) => setTimeout(r, 1200));
  //     setResultUri(image.uri);

  //     setPhase(PHASES.RESULT);
  //   } catch (e) {
  //     setError(e?.message || 'Classification failed');
  //     setPhase(PHASES.RESULT);
  //   }
  // }, [image, uriToBase64]);

  const retry = useCallback(() => {
    if (image?.uri) {
      setError(null);
      setPhase(PHASES.PREVIEW);
    } else {
      resetAll();
    }
  }, [image, resetAll]);

  // UI
  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>garbage classifier</Text>
        <Text style={styles.subtitle}>Capture → Preview → Classify</Text>

        {phase === PHASES.CAPTURE && (
          <Card>
            <SectionTitle>Capture Screen</SectionTitle>
            <Muted>Use camera or pick from gallery</Muted>
            <Row>
              <Btn label="Open Camera" onPress={openCamera} />
              <Btn label="Choose from Gallery" onPress={openGallery} ghost />
            </Row>
          </Card>
        )}

        {phase === PHASES.PREVIEW && (
          <Card>
            <SectionTitle>Preview</SectionTitle>
            {!!image?.uri && <Image style={styles.preview} source={{ uri: image.uri }} resizeMode="contain" />}
            <Muted>If ok — proceed to classification</Muted>
            <Row>
              <Btn label="Classify the item/items" onPress={classify} disabled={!canClassify} />
              <Btn label="Retake" onPress={resetAll} ghost />
            </Row>
          </Card>
        )}

        {phase === PHASES.PROCESSING && (
          <Card>
            <SectionTitle>Processing</SectionTitle>
            <ActivityIndicator size="large" color={COLORS.emerald} />
            <Muted>Please wait while we classify…</Muted>
          </Card>
        )}

        {phase === PHASES.RESULT && (
          <Card>
            <SectionTitle>Results</SectionTitle>
            {error ? (
              <>
                <Text style={[styles.muted, { color: COLORS.danger }]}>{String(error)}</Text>
                <Row>
                  <Btn label="Retry" onPress={retry} />
                  <Btn label="Start over" onPress={resetAll} ghost />
                </Row>
              </>
            ) : (
              <>
                <Muted>Server response image:</Muted>
                {!!resultUri && <Image style={styles.preview} source={{ uri: resultUri }} resizeMode="contain" />}
                <Row>
                  <Btn label="Classify another" onPress={resetAll} />
                </Row>
              </>
            )}
          </Card>
        )}
      </View>
    </SafeAreaView>
  );
}

// UI helpers
function Card({ children }) {
  return <View style={styles.card}>{children}</View>;
}
function SectionTitle({ children }) {
  return <Text style={styles.sectionTitle}>{children}</Text>;
}
function Muted({ children }) {
  return <Text style={styles.muted}>{children}</Text>;
}
function Row({ children }) {
  return <View style={styles.row}>{children}</View>;
}
function Btn({ label, onPress, ghost = false, disabled = false }) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.btn,
        ghost ? styles.btnGhost : styles.btnSolid,
        disabled && { opacity: 0.5 },
        pressed && { transform: [{ scale: 0.98 }] },
      ]}
    >
      <Text style={ghost ? styles.btnGhostText : styles.btnText}>{label}</Text>
    </Pressable>
  );
}

// Стили
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: COLORS.bg },
  container: { flex: 1, padding: 16, gap: 12 },
  title: { color: COLORS.fg, fontSize: 24, fontWeight: '700', letterSpacing: 0.3 },
  subtitle: { color: COLORS.gray, fontSize: 13, marginTop: -4 },
  card: {
    flex: 1,
    backgroundColor: COLORS.card,
    borderRadius: 14,
    padding: 16,
    gap: 12,
    borderWidth: 1,
    borderColor: '#1F2937',
    justifyContent: 'flex-start',
  },
  sectionTitle: { color: COLORS.fg, fontSize: 18, fontWeight: '600' },
  muted: { color: COLORS.gray, fontSize: 13 },
  row: { flexDirection: 'row', gap: 10, marginTop: 8, flexWrap: 'wrap' },
  btn: {
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    minWidth: 170,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnSolid: { backgroundColor: COLORS.emerald },
  btnGhost: { borderWidth: 1, borderColor: COLORS.emeraldDark },
  btnText: { color: '#052e22', fontWeight: '700' },
  btnGhostText: { color: COLORS.emerald },
  preview: {
    width: '100%',
    height: 360,
    backgroundColor: '#0b0f0e',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1F2937',
  },
});
