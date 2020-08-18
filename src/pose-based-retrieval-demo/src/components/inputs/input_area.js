import React from "react"
import {Container, Row, Col} from 'react-bootstrap'
import Button from 'react-bootstrap/Button'
import axios from 'axios';

import InputImg from "./input_img.js"
import ImgDisplay from "../displays/img_display.js"

import {decodeBase64} from '../../lib/utils.js'
import "./styles/input_styles.css"

class InputArea extends React.Component{
  constructor(props){
    super(props)
    this.state = {
        file: undefined,
        file_blob: undefined,
        file_url: "",
        file_name: "",
        display_name: ""
    }
    this.update_state = this.update_state.bind(this)
    this.startProcessing = this.startProcessing.bind(this)
    this.post_data = this.post_data.bind(this)
    this.get_disp = this.get_disp.bind(this)
  }

  // method that updates the state when a child component is changed
  async update_state(state_id, value){
    await this.setState({
      [state_id]:value
    });
  }

  // final checks before calling the API
  startProcessing(){
    // handling errors and exceptions
    if( this.state.file===undefined || this.state.file_name.length===0){
      return
    }
    this.post_data()
  }

  // sending image to the API for processing
  async post_data(){
    // creating an object to send to API via pose
    const formData = new FormData()
    const skip = ["file_blob"]
    formData.append("timestamp", new Date().toLocaleString())
    for (let name in this.state) {
      if(skip.includes(name)){
        continue
      }
      formData.append(name, this.state[name]);
    }

    // establishing connection, sendinng and awaiting response
    let url_object = undefined
    let results = undefined
    axios({
      method: 'post',
      url: 'http://localhost:5000/api/upload/',
      data: formData,
      headers: {'content-type': 'multipart/form-data',
                "Accept": "application/json"}
    })
    .then(function (response) {
        //handle success
        results = response
        let img_binary = results.data.data_binary
        url_object = decodeBase64(img_binary)
    })
    .catch(function (response) {
        //handle error
        results = 0
    })
    .finally(() => {
      // logic executed after having received the response
      if(results !== 0){
        let time = new Date().getTime()
        this.setState({
          file: url_object,
          file_blob: url_object,
          file_url: results.data.data_url + "?" + time,
          display_name: "Detections"
        })
      }
    });

  }

  // selects which image is goint ot be displayed on canvas
  get_disp(){
    if(this.state.file === this.state.file_blob){
      return this.state.file_blob
    }else{
      return this.state.file
    }
  }

  render(){
    var disp = this.get_disp()
    return(
      <Container className="input_area">
        <Row fluid="true">
          <Col md={2}>
          </Col>
          <Col sm={12} md={8}>
            <Row fluid="true">
              <ImgDisplay file={disp} file_url={this.state.file_url}
                          file_name={this.state.display_name}/>
            </Row>
            <Row className="buttons_area">
              <Col sm={1} md={1}></Col>
              <Col sm={4} md={4}>
                <InputImg className="myButton" update_state={this.update_state}/>
              </Col>
              <Col sm={1} md={2}></Col>
              <Col sm={4} md={4}>
                <Button className="myButton" variant="primary" onClick={this.startProcessing}>Process Image</Button>
              </Col>
              <Col sm={1} md={1}></Col>
            </Row>
          </Col>
          <Col md={2}>
          </Col>
        </Row>
      </Container>
    )
  }

}

export default InputArea

//
