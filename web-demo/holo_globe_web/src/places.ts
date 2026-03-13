export type PlaceRecord = {
  id: string;
  name: string;
  region: string;
  lat: number;
  lon: number;
  accent: string;
  description: string;
};

export const PLACES: PlaceRecord[] = [
  {
    id: "bucharest",
    name: "Bucharest",
    region: "Romania",
    lat: 44.4268,
    lon: 26.1025,
    accent: "#ffd271",
    description:
      "Dense urban command node with strong cultural signal, transit pressure, and a useful mix of civic and creative data layers.",
  },
  {
    id: "reykjavik",
    name: "Reykjavik",
    region: "Iceland",
    lat: 64.1466,
    lon: -21.9426,
    accent: "#8fe0ff",
    description:
      "High-latitude target for aurora-linked overlays, atmospheric sensing, and remote-environment storytelling.",
  },
  {
    id: "tokyo",
    name: "Tokyo",
    region: "Japan",
    lat: 35.6764,
    lon: 139.65,
    accent: "#ff9aa2",
    description:
      "High-density urban mesh ideal for mobility, pedestrian rhythm, and fast-changing interaction narratives.",
  },
  {
    id: "nairobi",
    name: "Nairobi",
    region: "Kenya",
    lat: -1.2921,
    lon: 36.8219,
    accent: "#7ff0bf",
    description:
      "East Africa relay connecting urban intelligence with wildlife corridors, climate overlays, and conservation stories.",
  },
  {
    id: "sao-paulo",
    name: "Sao Paulo",
    region: "Brazil",
    lat: -23.5505,
    lon: -46.6333,
    accent: "#7fd4ff",
    description:
      "Southern hemisphere mega-node suited for cultural event streams, logistics, and metropolitan pulse analysis.",
  },
  {
    id: "san-francisco",
    name: "San Francisco",
    region: "USA",
    lat: 37.7749,
    lon: -122.4194,
    accent: "#b39cff",
    description:
      "Prototype launch zone for XR systems, coastal sensing, and high-iteration spatial interface experiments.",
  },
];
