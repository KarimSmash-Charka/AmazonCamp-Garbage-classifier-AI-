const API_URL = 'http://192.168.1.142:8000/classify-image';

import React, { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  Platform,
} from 'react-native';

import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system/legacy';
import * as Base64 from 'base64-js';

import { LinearGradient } from 'expo-linear-gradient';
import { useFonts, Poppins_400Regular, Poppins_600SemiBold, Poppins_700Bold } from '@expo-google-fonts/poppins';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';


const COLORS = {
  bg1: '#0f2e26', 
  bg2: '#0c3b2e',
  bg3: '#115e59', 
  mint: '#34d399',
  lime: '#a3e635',
  cream: '#ecfdf5',
  card: 'rgba(255,255,255,0.06)',
  stroke: 'rgba(255,255,255,0.12)',
  danger: '#ef4444',
  text: '#e6fff5',
  subtext: 'rgba(230,255,245,0.7)',
};

const PHASES = {
  CAPTURE: 'CAPTURE',
  PREVIEW: 'PREVIEW',
  PROCESSING: 'PROCESSING',
  RESULT: 'RESULT',
};

export default function App() {

  const [fontsLoaded] = useFonts({
    Poppins_400Regular,
    Poppins_600SemiBold,
    Poppins_700Bold,
  });

  const [phase, setPhase] = useState(PHASES.CAPTURE);
  const [image, setImage] = useState(null);      
  const [resultUri, setResultUri] = useState(null);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState(null);

  const canClassify = useMemo(() => !!image && phase === PHASES.PREVIEW, [image, phase]);


  const ensurePermissions = useCallback(async () => {
    const lib = await ImagePicker.requestMediaLibraryPermissionsAsync();
    const cam = await ImagePicker.requestCameraPermissionsAsync();
    const ok = lib.status === 'granted' && cam.status === 'granted';
    if (!ok) Alert.alert('Permissions', 'Разрешите доступ к камере и галерее.');
    return ok;
  }, []);

  
  const openCamera = useCallback(async () => {
    try {
      if (!(await ensurePermissions())) return;
      const res = await ImagePicker.launchCameraAsync({
        quality: 0.9,
        base64: true,
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


  const base64ToBytes = (b64) => Base64.toByteArray(b64);
  const bytesToBase64 = (bytes) => Base64.fromByteArray(bytes);

  const uriToBase64 = useCallback(async (uri) => {
    const resp = await fetch(uri);
    const buf = new Uint8Array(await resp.arrayBuffer());
    return bytesToBase64(buf);
  }, []);

  const resetAll = useCallback(() => {
    setPhase(PHASES.CAPTURE);
    setImage(null);
    setResultUri(null);
    setDetections(null);
    setError(null);
  }, []);


  const classify = useCallback(async () => {
    if (!image?.uri) return;
    setPhase(PHASES.PROCESSING);
    setError(null);
    setResultUri(null);
    setDetections(null);

    try {
      const form = new FormData();
      form.append('file', { uri: image.uri, name: 'photo.jpg', type: 'image/jpeg' });

      const resp = await fetch(API_URL, {
        method: 'POST',
        body: form,
        headers: { Accept: 'image/jpeg' },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const detHeader = resp.headers.get('x-detections');
      if (detHeader) {
        try { setDetections(JSON.parse(detHeader)); } catch { setDetections([]); }
      }

      const arrBuf = await resp.arrayBuffer();
      const bytes = new Uint8Array(arrBuf);
      const b64 = Base64.fromByteArray(bytes);

      const outPath = FileSystem.cacheDirectory + `gc-result-${Date.now()}.jpg`;
      await FileSystem.writeAsStringAsync(outPath, b64, {
        encoding: (FileSystem?.EncodingType?.Base64) ?? 'base64',
      });

      setResultUri(outPath);
      setPhase(PHASES.RESULT);
    } catch (e) {
      console.log('fetch error', e);
      setError(e?.message || 'Classification failed');
      setPhase(PHASES.RESULT);
    }
  }, [image]);

  const retry = useCallback(() => {
    if (image?.uri) {
      setError(null);
      setPhase(PHASES.PREVIEW);
    } else {
      resetAll();
    }
  }, [image, resetAll]);

  if (!fontsLoaded) {
    return (
      <View style={[StyleSheet.absoluteFill, { alignItems: 'center', justifyContent: 'center', backgroundColor: COLORS.bg2 }]}>
        <ActivityIndicator size="large" />
      </View>
    );
  }


  return (
    <LinearGradient
      colors={[COLORS.bg1, COLORS.bg2, COLORS.bg3]}
      style={styles.gradient}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <SafeAreaView style={styles.safe}>

        <View style={styles.header}>
          <View style={styles.brandRow}>
            <MaterialCommunityIcons name="recycle-variant" size={26} color={COLORS.lime} />
            <Text style={styles.brand}>garbage classifier</Text>
          </View>
          <Text style={styles.tagline}>Sort smart. Help the planet.</Text>
        </View>

        <View style={styles.container}>

          <Card>
            {phase === PHASES.CAPTURE && (
              <>
                <SectionTitle icon="camera">
                  Capture or Pick an item
                </SectionTitle>
                <Muted>Take a photo or choose from gallery to identify recyclable category.</Muted>
                <Row center>
                  <Btn
                    label="Open Camera"
                    onPress={openCamera}
                    icon={<Feather name="camera" size={18} color="#052e22" />}
                  />
                  <Btn
                    label="Choose from Gallery"
                    onPress={openGallery}
                    ghost
                    icon={<Feather name="image" size={18} color={COLORS.mint} />}
                  />
                </Row>
              </>
            )}

            {phase === PHASES.PREVIEW && (
              <>
                <SectionTitle icon="image-multiple">
                  Preview
                </SectionTitle>
                {!!image?.uri && (
                  <View style={styles.previewWrap}>
                    <Image style={styles.preview} source={{ uri: image.uri }} resizeMode="contain" />
                  </View>
                )}
                <Muted>Happy with the shot? Classify it to see results.</Muted>
                <Row center>
                  <Btn
                    label="Classify the item/items"
                    onPress={classify}
                    disabled={!canClassify}
                    icon={<Feather name="play-circle" size={18} color="#052e22" />}
                  />
                  <Btn
                    label="Retake"
                    onPress={resetAll}
                    ghost
                    icon={<Feather name="rotate-ccw" size={18} color={COLORS.mint} />}
                  />
                </Row>
              </>
            )}

            {phase === PHASES.PROCESSING && (
              <>
                <SectionTitle icon="progress-clock">
                  Processing
                </SectionTitle>
                <ActivityIndicator size="large" />
                <Muted>Please wait while we classify…</Muted>
              </>
            )}

            {phase === PHASES.RESULT && (
              <>
                <SectionTitle icon="check-decagram">
                  Result
                </SectionTitle>

                {error ? (
                  <>
                    <Text style={[styles.muted, { color: COLORS.danger, fontFamily: 'Poppins_600SemiBold' }]}>
                      {String(error)}
                    </Text>
                    <Row center>
                      <Btn label="Retry" onPress={retry} icon={<Feather name="repeat" size={18} color="#052e22" />} />
                      <Btn label="Start over" onPress={resetAll} ghost icon={<Feather name="trash-2" size={18} color={COLORS.mint} />} />
                    </Row>
                  </>
                ) : (
                  <>
                    <Muted>Annotated image:</Muted>
                    {!!resultUri && (
                      <View style={styles.previewWrap}>
                        <Image style={styles.preview} source={{ uri: resultUri }} resizeMode="contain" />
                      </View>
                    )}


                    {!error && detections && detections.length > 0 && (
                      <View style={styles.detsBlock}>
                        <Muted style={{ marginBottom: 6 }}>Detections:</Muted>
                        {detections.map((d, i) => (
                          <View key={i} style={styles.detRow}>
                            <MaterialCommunityIcons name="leaf" size={16} color={COLORS.lime} />
                            <Text style={styles.detText}>
                              #{i + 1} [{d.bbox.join(', ')}] — {d.class} ({(d.class_conf * 100).toFixed(1)}%)
                            </Text>
                          </View>
                        ))}
                      </View>
                    )}

                    <Row center>
                      <Btn label="Classify another" onPress={resetAll} icon={<Feather name="plus-circle" size={18} color="#052e22" />} />
                    </Row>
                  </>
                )}
              </>
            )}
          </Card>


          <EcoFooter />
        </View>
      </SafeAreaView>
    </LinearGradient>
  );
}


function Card({ children }) {
  return <View style={styles.card}>{children}</View>;
}

function SectionTitle({ children, icon = 'leaf' }) {
  return (
    <View style={styles.sectionTitleRow}>
      <MaterialCommunityIcons name={icon} size={20} color={COLORS.lime} />
      <Text style={styles.sectionTitle}>{children}</Text>
    </View>
  );
}

function Muted({ children, style }) {
  return <Text style={[styles.muted, style]}>{children}</Text>;
}

function Row({ children, center = false }) {
  return <View style={[styles.row, center && { justifyContent: 'center' }]}>{children}</View>;
}

function Btn({ label, onPress, ghost = false, disabled = false, icon = null }) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.btn,
        ghost ? styles.btnGhost : styles.btnSolid,
        disabled && { opacity: 0.4 },
        pressed && { transform: [{ scale: 0.98 }] },
      ]}
    >
      <View style={styles.btnInner}>
        {icon && <View style={{ marginRight: 8 }}>{icon}</View>}
        <Text style={ghost ? styles.btnGhostText : styles.btnText}>{label}</Text>
      </View>
    </Pressable>
  );
}

function EcoFooter() {
  return (
    <View style={styles.footer}>
      <MaterialCommunityIcons name="recycle" size={18} color={COLORS.mint} />
      <Text style={styles.footerText}>Recycle today, breathe better tomorrow.</Text>
    </View>
  );
}


const styles = StyleSheet.create({
  gradient: { flex: 1 },
  safe: { flex: 1 },
  header: {
    paddingHorizontal: 18,
    paddingTop: Platform.OS === 'android' ? 14 : 6,
    paddingBottom: 4,
    marginTop: 60,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  brand: {
    fontFamily: 'Poppins_700Bold',
    fontSize: 22,
    color: COLORS.text,
    letterSpacing: 0.4,
    textTransform: 'lowercase',
  },
  tagline: {
    fontFamily: 'Poppins_400Regular',
    color: COLORS.subtext,
    marginTop: 2,
    fontSize: 12,
  },

  container: {
    flex: 1,
    paddingHorizontal: 16,
    paddingBottom: 12,
    justifyContent: 'center',
  },

  card: {
    backgroundColor: COLORS.card,
    borderRadius: 22,
    padding: 16,
    gap: 12,
    borderWidth: 1,
    borderColor: COLORS.stroke,
    shadowColor: '#000',
    shadowOpacity: 0.35,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 8 },
    elevation: 8,
  },

  sectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  sectionTitle: {
    color: COLORS.text,
    fontSize: 18,
    fontFamily: 'Poppins_600SemiBold',
  },
  muted: {
    color: COLORS.subtext,
    fontSize: 13,
    fontFamily: 'Poppins_400Regular',
  },

  row: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 8,
    flexWrap: 'wrap',
  },

  btn: {
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 18,
    minWidth: 180,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnInner: { flexDirection: 'row', alignItems: 'center' },
  btnSolid: {
    backgroundColor: COLORS.mint,
  },
  btnGhost: {
    borderWidth: 1,
    borderColor: COLORS.mint,
    backgroundColor: 'transparent',
  },
  btnText: {
    color: '#052e22',
    fontFamily: 'Poppins_700Bold',
    fontSize: 14,
  },
  btnGhostText: {
    color: COLORS.mint,
    fontFamily: 'Poppins_700Bold',
    fontSize: 14,
  },

  previewWrap: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.stroke,
    overflow: 'hidden',
    backgroundColor: 'rgba(0,0,0,0.25)',
  },
  preview: {
    width: '100%',
    height: 360,
  },

  detsBlock: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: 1,
    borderColor: COLORS.stroke,
    borderRadius: 14,
    padding: 10,
    marginTop: 6,
    gap: 6,
  },
  detRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  detText: {
    color: COLORS.text,
    fontSize: 13,
    fontFamily: 'Poppins_400Regular',
  },

  footer: {
    alignSelf: 'center',
    marginTop: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    opacity: 0.9,
  },
  footerText: {
    color: COLORS.cream,
    fontFamily: 'Poppins_600SemiBold',
    fontSize: 12,
    letterSpacing: 0.3,
  },
});
