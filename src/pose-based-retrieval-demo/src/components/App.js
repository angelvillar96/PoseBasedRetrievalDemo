import React from 'react';
import './App.css';

// self defined components
import Title from './titles/title.js'
import DropFile from "./inputs/input_img"

function App() {
  return (
    <div className="App">
      <header className="App-header">
      </header>
      <Title/>
      <DropFile/>
    </div>
  );
}

export default App;
