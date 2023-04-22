import React from "react"
import {Container, Row, Col} from 'react-bootstrap'
import {Button} from 'react-bootstrap'
import axios from 'axios';

import InputImg from "./input_img.js"
import TakePicture from "./take_picture.js"
import AdvancedSettings from "./advanced_settings.js"
import ImgDisplay from "../displays/img_display.js"
import DetDisplay from "../displays/det_display.js"
import ArrowButton from "../../../components/arrow_button/arrow_button.js"

import {decodeBase64} from '../../lib/utils.js'
import "./styles/input_styles.css"

class InputArea extends React.Component{
  constructor(props){
    super(props)
    this.state = {
        // files and image urls
        file: undefined,
        file_blob: undefined,
        file_url: "",
        file_name: "",
        display_name: "",
        // instance and pose images
        dets: "",
        poses: "",
        // vectors used for retrieval
        pose_vectors: "",
        keypoints: "",
        // retrieval_properties
        person_detector: "Tuned R-CNN",
        keypoint_detector: "Baseline HRNet",
        retrieval_method: "Approx. kNN",
        retrieval_database: "Christian Arch",
        // auxiliar state variables
        lim: 0,  // index of detections to display, acutalizes with arrows
    }
    // method used for updating the results. Comes as a prop from the Root App component
    // this.update_results = this.props.update_results.bind(this)
    this.update_results = this.props.update_results

    this.update_state = this.update_state.bind(this)
    this.startProcessing = this.startProcessing.bind(this)
    this.post_data = this.post_data.bind(this)
    this.get_disp = this.get_disp.bind(this)
    this.next_dets = this.next_dets.bind(this)
    this.previous_dets = this.previous_dets.bind(this)
    this.get_retrieval_settings = this.get_retrieval_settings.bind(this)
  }

  // method that updates the state when a child component is changed
  async update_state(state_id, value){
    await this.setState({
      [state_id]:value
    });
  }

  // method for reading the retrieval settings from a child component
  get_retrieval_settings(){
    var retrieval_settings = {
      "person_detector": this.state.person_detector,
      "keypoint_detector": this.state.keypoint_detector,
      "retrieval_method": this.state.retrieval_method,
      "retrieval_database": this.state.retrieval_database
    }
    return retrieval_settings
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
    for (var name in this.state) {
      if(skip.includes(name)){
        continue
      }
      formData.append(name, this.state[name]);
    }

    var url_object = undefined
    var detections = undefined
    var poses = undefined
    var results = undefined
    var $this = this
    // establishing connection, sending and awaiting response
    axios({
      method: 'post',
       url: 'http://localhost:5000/api/upload/',
//       url: "http://131.188.35.240:5000/api/upload/",
//      url: 'http://131.188.35.181:5000/api/upload/',
      data: formData,
      headers: {'content-type': 'multipart/form-data',
                "Accept": "application/json"}
    })
    .then(function (response) {
        //handle success
        console.log("Success")
        results = response
        var img_binary = results.data.img_binary
        url_object = decodeBase64(img_binary)
        poses = []
        detections = []
        for(var i=0; i<results.data.poses.length; i++){
          var cur_pose = decodeBase64(results.data.poses[i])
          poses.push(cur_pose)
          var cur_det = decodeBase64(results.data.detections[i])
          detections.push(cur_det)
        }
        // cleaning the results section
        var clean_retrievals = {
          images: [],
          metadata: []
        }
        $this.update_results(clean_retrievals)
    })
    .catch(function (response) {
        //handle error
        console.log("Error!")
        results = 0
    })
    .finally(() => {
      // logic executed after having received the response
      console.log("Updating state")
      if(results !== 0){
        var time = new Date().getTime()
        this.setState({
          file: url_object,
          file_blob: url_object,
          file_url: results.data.data_url + "?" + time,
          poses: poses,
          dets: detections,
          pose_vectors: results.data.pose_vectors,
          keypoints: results.data.keypoints,
          display_name: "Detections"
        })
      }
    });

  }

  // processing click on arrows for previous and next dets
  next_dets(){
    var next_lim = this.state.lim + 2;
    if(next_lim >= this.state.poses.length){
      next_lim = 0
    }
    this.setState({
      lim: next_lim
    })
  }
  previous_dets(){
    var next_lim = this.state.lim - 2;
    if(next_lim < 0){
      next_lim = this.state.poses.length - 3
    }
    this.setState({
      lim: next_lim
    })
  }


  // selects which image is going ot be displayed on canvas: original or requested
  get_disp(){
    if(this.state.file === this.state.file_blob){
      return this.state.file_blob
    }else{
      return this.state.file
    }
  }

  render(){

    // creating a display box for each detected instance
    var disp = this.get_disp()
    var det_displays = []
    var cur_dets_disp = []
    var leftArrow = undefined
    var rightArrow = undefined
    for(var i=0; i<this.state.poses.length; i++){
      leftArrow =  <ArrowButton orientation="left" onClick={this.previous_dets}/>
      rightArrow =  <ArrowButton orientation="right" onClick={this.next_dets}/>
      var cur_det_display = {
        id:i,
        value: <DetDisplay file={this.state.poses[i]} pose_vector={this.state.pose_vectors[i]}
                           keypoints={this.state.keypoints} det_idx={i+1}
                           update_results={this.update_results}
                           get_retrieval_settings={this.get_retrieval_settings}/>
      }
      det_displays.push(cur_det_display)
      if((i >= this.state.lim) && (i < this.state.lim + 2)){
        cur_dets_disp.push(cur_det_display)
      }
    }


    return(
      <div className="input_area">
        <Row fluid="true">
          <Col md={12}>
            <Row fluid="true">
              <ImgDisplay file={disp} file_url={this.state.file_url}
                          file_name={this.state.display_name} disp_type="img_display"/>
            </Row>
            <Row className="buttons_area">
              <Col sm={1} md={1}></Col>
              <Col sm={4} md={4}>
                <InputImg className="myButton" update_state={this.update_state}/>
              </Col>
              <Col sm={1} md={2}></Col>
              <Col sm={4} md={4}>
                <TakePicture className="myButton" update_state={this.update_state}/>
              </Col>
              <Col sm={1} md={1}></Col>
            </Row>
            <Row>
              <AdvancedSettings update_state={this.update_state}/>
            </Row>
            <Row className="buttons_area process_button_row">
              <Col sm={3} md={4}></Col>
              <Col sm={6} md={4}>
                <Button className="myButton" variant="primary"
                        onClick={this.startProcessing}>
                  Process Image
                </Button>
              </Col>
              <Col sm={3} md={4}></Col>
            </Row>
          </Col>
        </Row>
        <Row className="detsArea">
          <Col md={1} style={{margin:"auto"}}>
            {leftArrow}
          </Col>
          {cur_dets_disp.map(cur_det => (
            <Col md={5} lg={5} key={cur_det.id}>{cur_det.value}</Col>
          ))}
          <Col md={1} style={{margin:"auto"}}>
            {rightArrow}
          </Col>
        </Row>
      </div>
    )
  }

}

export default InputArea

//
