import { View, Text, TouchableOpacity } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

const COLORS = {
  bg: "#0B0F14",
  card: "#121722",
  border: "#223043",
  text: "#E8EEF8",
  subt: "#A5B0C4",
  primary: "#3A75FF",
  danger: "#FF3B3B",
  success: "#27C499",
  warn: "#FFB020",
};

function badgeColor(diagnosis: string) {
  if (/likely/i.test(diagnosis)) return COLORS.danger;
  if (/uncertain/i.test(diagnosis)) return COLORS.warn;
  return COLORS.success;
}

export default function Results() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const probability =
    Math.round(parseFloat((params.probability as string) || "0") * 100) / 1; // already in 0..100? backend is 0..1
  // Your backend returns 0..1; convert to percent:
  const percent = Math.round(
    parseFloat((params.probability as string) || "0") * 100 + Number.EPSILON
  );

  const diagnosis = (params.diagnosis as string) || "Uncertain";
  const confidence = (params.confidence as string) || "moderate";

  const Fo = params.Fo as string | undefined;
  const Jitter = params.Jitter as string | undefined;
  const HNR = params.HNR as string | undefined;

  return (
    <View
      style={{
        flex: 1,
        backgroundColor: COLORS.bg,
        padding: 20,
      }}
    >
      <View
        style={{
          backgroundColor: COLORS.card,
          borderWidth: 1,
          borderColor: COLORS.border,
          borderRadius: 18,
          padding: 18,
          gap: 12,
          marginTop: 50,
        }}
      >
        <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
          <Ionicons name="ribbon" size={20} color={COLORS.primary} />
          <Text style={{ color: COLORS.text, fontSize: 18, fontWeight: "700" }}>
            Voice Analysis Result
          </Text>
        </View>

        {/* Big % */}
        <View style={{ alignItems: "center", paddingVertical: 8 }}>
          <Text style={{ color: COLORS.text, fontSize: 56, fontWeight: "800" }}>
            {percent}%
          </Text>
          <Text style={{ color: COLORS.subt, marginTop: -6 }}>
            Parkinson’s probability
          </Text>
        </View>

        {/* Diagnosis pill */}
        <View
          style={{
            alignSelf: "center",
            paddingVertical: 6,
            paddingHorizontal: 12,
            backgroundColor: badgeColor(diagnosis),
            borderRadius: 999,
          }}
        >
          <Text
            style={{ color: "white", fontWeight: "700", letterSpacing: 0.3 }}
          >
            {diagnosis}
          </Text>
        </View>

        <Text style={{ color: COLORS.subt, marginTop: 8, textAlign: "center" }}>
          Confidence: {confidence}. This is a screening tool, not a diagnosis.
        </Text>
      </View>

      {/* Feature highlights (optional) */}
      <View style={{ marginTop: 14, flexDirection: "row", gap: 10 }}>
        {Fo && (
          <View
            style={{
              flex: 1,
              backgroundColor: COLORS.card,
              borderColor: COLORS.border,
              borderWidth: 1,
              borderRadius: 14,
              padding: 12,
              gap: 4,
            }}
          >
            <Text style={{ color: COLORS.subt, fontSize: 12 }}>Fo (Hz)</Text>
            <Text
              style={{ color: COLORS.text, fontWeight: "700", fontSize: 16 }}
            >
              {Fo}
            </Text>
          </View>
        )}
        {Jitter && (
          <View
            style={{
              flex: 1,
              backgroundColor: COLORS.card,
              borderColor: COLORS.border,
              borderWidth: 1,
              borderRadius: 14,
              padding: 12,
              gap: 4,
            }}
          >
            <Text style={{ color: COLORS.subt, fontSize: 12 }}>Jitter (%)</Text>
            <Text
              style={{ color: COLORS.text, fontWeight: "700", fontSize: 16 }}
            >
              {Jitter}
            </Text>
          </View>
        )}
        {HNR && (
          <View
            style={{
              flex: 1,
              backgroundColor: COLORS.card,
              borderColor: COLORS.border,
              borderWidth: 1,
              borderRadius: 14,
              padding: 12,
              gap: 4,
            }}
          >
            <Text style={{ color: COLORS.subt, fontSize: 12 }}>HNR (dB)</Text>
            <Text
              style={{ color: COLORS.text, fontWeight: "700", fontSize: 16 }}
            >
              {HNR}
            </Text>
          </View>
        )}
      </View>

      {/* CTA buttons */}
      <View style={{ marginTop: 18, flexDirection: "row", gap: 10 }}>
        <TouchableOpacity
          onPress={() => router.push("/(tabs)/record")}
          style={{
            flex: 1,
            backgroundColor: COLORS.primary,
            borderRadius: 14,
            paddingVertical: 14,
            alignItems: "center",
          }}
        >
          <Text style={{ color: "white", fontWeight: "700" }}>Test Again</Text>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={() => router.replace("/(tabs)/home")}
          style={{
            flex: 1,
            backgroundColor: COLORS.card,
            borderColor: COLORS.border,
            borderWidth: 1,
            borderRadius: 14,
            paddingVertical: 14,
            alignItems: "center",
          }}
        >
          <Text style={{ color: COLORS.text, fontWeight: "700" }}>Go Home</Text>
        </TouchableOpacity>
      </View>

      <View style={{ marginTop: 10 }}>
        <Text style={{ color: COLORS.subt, fontSize: 12, textAlign: "center" }}>
          Try recording in a quiet room at steady volume for clearer results.
        </Text>
      </View>
    </View>
  );
}
