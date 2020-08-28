import React from 'react';
import './App.css';

// self defined components
import Title from './titles/title.js'
import Text from './titles/text_display'
import Separator from './titles/separator.js'
import InputArea from "./inputs/input_area.js"
import ResultsArea from "./results/results_area.js"

class App extends React.Component{

  constructor(props){
    super(props)
    this.state = {
      images: [],
      metadata: []
    }
    this.update_results = this.update_results.bind(this)
  }

  // receiving the new results
  async update_results(results){
    await this.setState({
      images: results.images,
      metadata: results.metadata
    })
    console.log("Updated results! :)")
  }

  render(){
    return (
      <div className="App">
        <header className="App-header">
        </header>
        <Title/>
        <Separator title="Input"/>
        <InputArea update_results={this.update_results}/>
        <Separator title="Results"/>
        <ResultsArea images={this.state.images} metadata={this.state.metadata}/>
        <Separator/>
        <Text text_display="by Angel Villar-Corrales" font_size="1.3em"/>
        <Separator/>
      </div>
    );
  }
}

export default App;
