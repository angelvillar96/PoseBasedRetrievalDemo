import React from "react"
import {Container, Row, Col} from 'react-bootstrap'
import Button from 'react-bootstrap/Button'
import axios from 'axios';

import InputImg from "./input_img.js"
import ImgDisplay from "../displays/img_display.js"

import "./styles/input_styles.css"

class InputArea extends React.Component{
  constructor(props){
    super(props)
    this.state = {
        file: undefined,
        file_url: "",
        file_name: ""
    }
    this.update_state = this.update_state.bind(this)
    this.startProcessing = this.startProcessing.bind(this)
    this.post_data = this.post_data.bind(this)
  }

  // method that updates the state when a child component is changed
  update_state(state_id, value){
    this.setState({
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
  post_data(){
    // sending files to the server
    const formData = new FormData()
    formData.append("timestamp", new Date().toLocaleString())
    for (let name in this.state) {
      formData.append(name, this.state[name]);
    }

    axios({
      method: 'post',
      url: 'http://localhost:5000/api/upload',
      data: formData,
      headers: {'content-type': 'multipart/form-data' }
    })
    .then(function (response) {
        //handle success
        console.log("Success!!")
        console.log(response);
    })
    .catch(function (response) {
        //handle error
        console.log("Error!!")
        console.log(response);
    });
  }


  render(){
    return(
      <Container className="input_area">
        <Row fluid="true">
          <Col md={2}>
          </Col>
          <Col sm={12} md={8}>
            <Row fluid="true">
              <ImgDisplay file={this.state.file} file_url={this.state.file_url}
                          file_name={this.state.file_name}/>
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
