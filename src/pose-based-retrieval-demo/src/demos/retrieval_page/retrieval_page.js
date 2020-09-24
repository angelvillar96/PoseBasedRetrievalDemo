import React from 'react';
import './retrieval_page.css';
import {Container, Row, Col} from 'react-bootstrap'

// self defined components
import Header from '../../components/pr_lab_header/title.js'
import Separator from '../../components/separator/separator.js'
import Text from './titles/text_display'
import InputArea from "./inputs/input_area.js"
import ResultsArea from "./results/results_area.js"

class RetrievalPage extends React.Component{

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
      <div>
        <Header title={"Pose-Based Image Retrieval Demo"}/>
        <Separator/>
        <Container className="App">
          <Row>
          </Row>
          <Row>
            <Col md={6}>
              <InputArea update_results={this.update_results}/>
            </Col>
            <Col md={6}>
              <ResultsArea images={this.state.images} metadata={this.state.metadata}/>
            </Col>
          </Row>
          <Row>
            <Separator/>
            <Col md={12}>
              <Text text_display="by Angel Villar-Corrales" font_size="1.3em"/>
            </Col>
          </Row>
        </Container>
      </div>
    );
  }
}

export default RetrievalPage;
