import React from 'react';
import './App.css';

// self defined components
import Title from './titles/title.js'
import Separator from './titles/separator.js'
import InputArea from "./inputs/input_area.js"

function App() {
  return (
    <div className="App">
      <header className="App-header">
      </header>
      <Title/>
      <Separator title="Input"/>
      <InputArea/>
      <Separator title="Results"/>
      <Separator/>
      <Separator/>
    </div>
  );
}

export default App;
