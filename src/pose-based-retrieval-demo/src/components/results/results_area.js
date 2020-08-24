import React from "react"
import {Container, Row, Col} from 'react-bootstrap'

import ResultDisplay from "../displays/result_display.js"

import {decodeBase64} from '../../lib/utils.js'
import "./styles/results_styles.css"

class ResultsArea extends React.Component{
  constructor(props){
    super(props)
    this.state = {
        images: this.props.images,
        metrics: this.props.metrics
    }
  }

  render(){

    // creating a display object for each of the retrievals
    var retrieval_displays = []
    for(var i=0; i<this.props.images.length; i++){
      var cur_retrieval = {
        id:i,
        value: <ResultDisplay file={decodeBase64(this.props.images[i])}
                              det_idx={i+1} metric={this.props.metrics[i]}/>
      }
      retrieval_displays.push(cur_retrieval)
    }

    return(
      <Container>
        <Row className="detsArea">
          {retrieval_displays.map(cur_retrieval => (
            <Col sm={4} md={3} key={cur_retrieval.id}>{cur_retrieval.value}</Col>
          ))}
        </Row>
      </Container>
    )
  }
}

export default ResultsArea

//
