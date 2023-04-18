import React from "react"
import {Container, Row, Col} from 'react-bootstrap'

import ResultDisplay from "../displays/result_display.js"
import ArrowButton from "../../../components/arrow_button/arrow_button.js"
import Text from '../titles/text_display'

import {decodeBase64} from '../../lib/utils.js'
import "./styles/results_styles.css"

class ResultsArea extends React.Component{
  constructor(props){
    super(props)
    this.state = {
        images: this.props.images,
        metrics: this.props.metrics,
        num_results: 6,
        lim: 0
      }

      this.next_results = this.next_results.bind(this)
      this.previous_results = this.previous_results.bind(this)
  }


  // processing click on arrows for previous and next results
  next_results(){
    console.log( this.props.images.length)
    var next_lim = (this.state.lim + this.state.num_results / 2);
    if(next_lim >= this.props.images.length){
      next_lim = 0
    }
    console.log("Updating lim from " +  this.state.lim + " to " + next_lim)
    this.setState({
      lim: next_lim
    })
  }
  previous_results(){
    var next_lim = this.state.lim - (this.state.num_results / 2);
    if(next_lim < 0){
      next_lim = this.props.images.length - (this.state.num_results / 2) - 1;
    }
    console.log("Updating lim from " +  this.state.lim + " to " + next_lim)
    this.setState({
      lim: next_lim
    })
  }

  render(){
    // creating a display object for each of the retrievals
    var retrieval_displays = []
    var cur_results_disp = []
    var leftArrow = undefined
    var rightArrow = undefined
    var title = ""

    console.log("Total retrieved images: " + this.props.images.length)

    for(var i=this.state.lim; i<this.state.num_results + this.state.lim; i++){
      if(i >= this.props.images.length){
        break
      }
      title = "Retrieval Results"
      leftArrow =  <ArrowButton orientation="left" onClick={this.previous_results}/>
      rightArrow =  <ArrowButton orientation="right" onClick={this.next_results}/>
      var cur_retrieval = {
        id:i,
        value: <ResultDisplay file={decodeBase64(this.props.images[i])}
                              det_idx={i+1} metric={this.props.metadata.distance[i]}/>
      }
      retrieval_displays.push(cur_retrieval)
      cur_results_disp.push(cur_retrieval)
    }


    return(
      <div>
        <Row className="resultsTitle">
          <Text text_display={title} font_size="1.4em" font_style="normal"/>
        </Row>
        <Row className="detsArea">
          <Col md={1} style={{margin:"auto"}}>
            {leftArrow}
          </Col>
          <Col sm={10} md={10}>
            <Row>
            {cur_results_disp.map(cur_result => (
              <Col sm={4} md={4} key={cur_result.id}>{cur_result.value}</Col>
            ))}
            </Row>
          </Col>
          <Col md={1} style={{margin:"auto"}}>
            {rightArrow}
          </Col>
        </Row>
      </div>
    )
  }
}

export default ResultsArea

//
