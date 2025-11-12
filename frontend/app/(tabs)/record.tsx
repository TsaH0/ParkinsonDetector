import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Animated,
} from "react-native";
import { Audio } from "expo-av";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

const COLORS = {
  bg: "#0B0F14",
  card: "#121722",
  border: "#223043",
  text: "#E8EEF8",
  subt: "#A5B0C4",
  primary: "#3A75FF",
  danger: "#FF3B3B",
  warn: "#FFB020",
};

const BACKEND_URL = "http://172.20.10.2:8000/predict"; // change if needed

export default function Record() {
  const router = useRouter();
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [err, setErr] = useState<string | null>(null);

  const scale = useRef(new Animated.Value(1)).current;
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startPulse = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(scale, {
          toValue: 1.12,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(scale, {
          toValue: 1.0,
          duration: 600,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const stopPulse = () => {
    scale.stopAnimation();
    scale.setValue(1);
  };

  const startTimer = () => {
    setSeconds(0);
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
  };

  const stopTimer = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  const startRecording = async () => {
    try {
      setErr(null);
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        setErr("Microphone permission is required.");
        return;
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const rec = new Audio.Recording();
      await rec.prepareToRecordAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      await rec.startAsync();

      setRecording(rec);
      startPulse();
      startTimer();
    } catch (error: any) {
      setErr(error?.message ?? "Recording failed to start.");
    }
  };

  const stopRecording = async () => {
    stopPulse();
    stopTimer();
    if (!recording) return;
    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      if (uri) {
        await uploadAudio(uri);
      }
    } catch (error: any) {
      setErr(error?.message ?? "Failed to stop recording.");
    }
  };

  const uploadAudio = async (uri: string) => {
    try {
      setIsLoading(true);
      const formData = new FormData();
      // Note: backend accepts wav or mp3; expo-av usually gives m4a.
      // Backend will read it fine; set type accordingly.
      formData.append("audio", {
        uri,
        name: "voice.m4a",
        type: "audio/m4a",
      } as any);

      const res = await fetch(BACKEND_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Server error (${res.status}): ${t}`);
      }
      const json = await res.json();

      router.push({
        pathname: "/(tabs)/results",
        params: {
          probability: json.probability?.toString() ?? "0.0",
          diagnosis: json.diagnosis ?? "Uncertain",
          confidence: json.confidence ?? "moderate",
          Fo: json.features?.Fo?.toString() ?? "",
          Jitter: json.features?.["Jitter(%)"]?.toString() ?? "",
          HNR: json.features?.HNR?.toString() ?? "",
        },
      });
    } catch (error: any) {
      setErr(error?.message ?? "Upload failed. Check network & backend.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      stopPulse();
      stopTimer();
    };
  }, []);

  return (
    <View
      style={{
        flex: 1,
        backgroundColor: COLORS.bg,
        justifyContent: "center",
        alignItems: "center",
        paddingHorizontal: 20,
      }}
    >
      <View
        style={{
          backgroundColor: COLORS.card,
          paddingVertical: 28,
          paddingHorizontal: 22,
          borderRadius: 16,
          width: "100%",
          alignItems: "center",
          borderWidth: 1,
          borderColor: COLORS.border,
          gap: 10,
        }}
      >
        <Text style={{ color: COLORS.text, fontSize: 18, fontWeight: "700" }}>
          {recording ? "Recording…" : "Ready to record"}
        </Text>

        <Text
          style={{
            color:
              seconds >= 8
                ? COLORS.danger // 8+ sec
                : seconds >= 5
                ? COLORS.warn // 5–8 sec
                : COLORS.subt, // 0–5 sec
            fontSize: 13,
            fontWeight: seconds >= 5 ? "600" : "400",
          }}
        >
          Hold “aaaaah” for 5–8 seconds • {seconds}s
        </Text>

        <Animated.View style={{ transform: [{ scale }], marginTop: 16 }}>
          <TouchableOpacity
            onPress={recording ? stopRecording : startRecording}
            style={{
              backgroundColor: recording ? COLORS.danger : COLORS.primary,
              width: 120,
              height: 120,
              borderRadius: 100,
              justifyContent: "center",
              alignItems: "center",
              shadowColor: "#000",
              shadowOpacity: 0.25,
              shadowRadius: 16,
              elevation: 6,
            }}
          >
            <Ionicons name="mic" color="white" size={48} />
          </TouchableOpacity>
        </Animated.View>

        <Text
          style={{ color: COLORS.subt, textAlign: "center", marginTop: 10 }}
        >
          Hold the vowel sound “aaaaah” continuously for{" "}
          <Text style={{ fontWeight: "700", color: COLORS.text }}>
            5–8 seconds
          </Text>
          .
        </Text>

        {!!err && (
          <View
            style={{
              marginTop: 8,
              backgroundColor: "rgba(255,59,59,0.12)",
              borderColor: "rgba(255,59,59,0.25)",
              borderWidth: 1,
              padding: 8,
              borderRadius: 10,
              width: "100%",
            }}
          >
            <Text style={{ color: "#FF9F9F" }}>{err}</Text>
          </View>
        )}
      </View>

      {isLoading && (
        <View
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.65)",
            justifyContent: "center",
            alignItems: "center",
            paddingHorizontal: 24,
          }}
        >
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={{ marginTop: 16, color: "white", fontSize: 16 }}>
            Analyzing your voice…
          </Text>
          <Text style={{ marginTop: 6, color: "#D0DAEE", fontSize: 12 }}>
            This may take a few seconds depending on network & model.
          </Text>
        </View>
      )}
    </View>
  );
}
