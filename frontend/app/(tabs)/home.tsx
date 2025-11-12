import { View, Text, TouchableOpacity, Image, Platform } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";

const COLORS = {
  bg: "#0B0F14",
  card: "#121722",
  glass: "rgba(255,255,255,0.06)",
  primary: "#3A75FF",
  text: "#E8EEF8",
  subt: "#A5B0C4",
  border: "#223043",
  success: "#27C499",
  warn: "#FFB020",
};

export default function Home() {
  const router = useRouter();

  return (
    <View style={{ flex: 1, backgroundColor: COLORS.bg }}>
      {/* Hero */}
      <LinearGradient
        colors={["#0E1420", "#0B0F14"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 0.8, y: 1 }}
        style={{
          paddingTop: 64,
          paddingHorizontal: 22,
          paddingBottom: 24,
          borderBottomWidth: 1,
          borderBottomColor: "rgba(255,255,255,0.06)",
        }}
      >
        <View
          style={{
            backgroundColor: COLORS.glass,
            borderWidth: 1,
            borderColor: "rgba(255,255,255,0.08)",
            borderRadius: 18,
            padding: 18,
          }}
        >
          <Text
            style={{
              color: COLORS.text,
              fontSize: 22,
              fontWeight: "700",
              marginBottom: 8,
            }}
          >
            Parkinson’s Voice Check
          </Text>
          <Text style={{ color: COLORS.subt }}>
            Hold the vowel “aaaaah” smoothly for{" "}
            <Text style={{ fontWeight: "700", color: COLORS.text }}>
              5–8 seconds
            </Text>
            .
          </Text>

          <TouchableOpacity
            onPress={() => router.push("/(tabs)/record")}
            style={{
              marginTop: 16,
              alignSelf: "flex-start",
              backgroundColor: COLORS.primary,
              borderRadius: 14,
              paddingVertical: 12,
              paddingHorizontal: 16,
              flexDirection: "row",
              gap: 8,
            }}
          >
            <Ionicons name="mic" size={18} color="white" />
            <Text
              style={{ color: "white", fontWeight: "700", letterSpacing: 0.3 }}
            >
              Start Voice Test
            </Text>
          </TouchableOpacity>
        </View>
      </LinearGradient>

      {/* Three tips cards */}
      <View
        style={{
          padding: 20,
          gap: 12,
        }}
      >
        <View
          style={{
            backgroundColor: COLORS.card,
            borderRadius: 16,
            padding: 14,
            borderWidth: 1,
            borderColor: COLORS.border,
            gap: 8,
          }}
        >
          <View style={{ flexDirection: "row", gap: 10, alignItems: "center" }}>
            <Ionicons name="volume-high" size={18} color={COLORS.success} />
            <Text style={{ color: COLORS.text, fontWeight: "600" }}>
              Tip: Clear, steady voice
            </Text>
          </View>
          <Text style={{ color: COLORS.subt }}>
            Keep the phone 10–15cm away. Don’t whisper. Avoid background noise.
          </Text>
        </View>

        <View
          style={{
            backgroundColor: COLORS.card,
            borderRadius: 16,
            padding: 14,
            borderWidth: 1,
            borderColor: COLORS.border,
            gap: 8,
          }}
        >
          <View style={{ flexDirection: "row", gap: 10, alignItems: "center" }}>
            <Ionicons name="timer" size={18} color={COLORS.warn} />
            <Text style={{ color: COLORS.text, fontWeight: "600" }}>
              3–5 seconds
            </Text>
          </View>
          <Text style={{ color: COLORS.subt }}>
            Hold the vowel “aaaaah” smoothly. We detect unstable sections.
          </Text>
        </View>

        <View
          style={{
            backgroundColor: COLORS.card,
            borderRadius: 16,
            padding: 14,
            borderWidth: 1,
            borderColor: COLORS.border,
            gap: 8,
          }}
        >
          <View style={{ flexDirection: "row", gap: 10, alignItems: "center" }}>
            <Ionicons
              name="shield-checkmark"
              size={18}
              color={COLORS.primary}
            />
            <Text style={{ color: COLORS.text, fontWeight: "600" }}>
              Not a diagnosis
            </Text>
          </View>
          <Text style={{ color: COLORS.subt }}>
            This is a screening tool. Consult a clinician for medical advice.
          </Text>
        </View>
      </View>
    </View>
  );
}
