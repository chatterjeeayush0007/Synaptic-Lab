/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

export default function App() {
  return <div></div>;
} createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);

