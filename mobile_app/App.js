import React from 'react';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { Feather } from '@expo/vector-icons';

import HomeScreen from './src/screens/HomeScreen';
import MapScreen from './src/screens/MapScreen';
import RecordScreen from './src/screens/RecordScreen';
import DiaryScreen from './src/screens/DiaryScreen';
import QuietSpotsScreen from './src/screens/QuietSpotsScreen';
import SoundHuntScreen from './src/screens/SoundHuntScreen';
import { COLORS } from './src/theme/colors';

const Tab = createBottomTabNavigator();

// One icon per tab so navigation reads at a glance instead of relying
// on text labels alone — Feather, matching the icon set already used
// throughout the app (RecordScreen, HomeScreen).
const TAB_ICONS = {
  Home: 'home',
  Analyse: 'activity',
  Map: 'map',
  Diary: 'book-open',
  'Quiet Spots': 'volume-1',
  'Sound Hunt': 'crosshair',
};

export default function App() {
  return (
    <>
      <StatusBar style="dark" />
      <NavigationContainer theme={{
        ...DefaultTheme,
        colors: {
          ...DefaultTheme.colors,
          background: COLORS.background,
          card: COLORS.background,
          border: COLORS.cardBorder,
          text: COLORS.text,
          primary: COLORS.primary,
        }
      }}>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            headerShown: false,
            tabBarActiveTintColor: COLORS.primary,
            tabBarInactiveTintColor: COLORS.textMuted,
            tabBarStyle: {
              backgroundColor: COLORS.surface,
              borderTopColor: COLORS.cardBorder,
            },
            tabBarLabelStyle: { fontSize: 10.5, fontWeight: '600' },
            tabBarIcon: ({ color, size }) => (
              <Feather name={TAB_ICONS[route.name]} size={size ? size - 2 : 20} color={color} />
            ),
          })}
        >
          <Tab.Screen name="Home" component={HomeScreen} />
          <Tab.Screen name="Analyse" component={RecordScreen} />
          <Tab.Screen name="Map" component={MapScreen} />
          <Tab.Screen name="Diary" component={DiaryScreen} />
          <Tab.Screen name="Quiet Spots" component={QuietSpotsScreen} />
          <Tab.Screen name="Sound Hunt" component={SoundHuntScreen} />
        </Tab.Navigator>
      </NavigationContainer>
    </>
  );
}
