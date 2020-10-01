  import React from "react"
import {Container, Row, Col} from 'react-bootstrap'
import Collapsible from 'react-collapsible';

import CustomDropdown from "../../../components/dropdown/dropdown.js"
import {BsPlusCircle} from "react-icons/bs"
import {AiOutlineMinusCircle} from "react-icons/ai"

import "./styles/input_styles.css"

class AdvancedSettings extends React.Component{


  constructor(props){
    super(props)
    this.state = {
      icon: <BsPlusCircle/>  // if True, expand is closed
    }

    this.open = this.open.bind(this)
    this.close = this.close.bind(this)
  }

  open(){
    this.setState({icon: <AiOutlineMinusCircle/>})
    return
  }
  close(){
    this.setState({icon: <BsPlusCircle/>})
  }

  render(){

    var title = <span>{this.state.icon}  Advanced Settings </span>

    const person_detectors = ["Faster R-CNN", "EfficientDet", "Tuned R-CNN"]
    const keypoint_detectors = ["Baseline HRNet", "Styled HRNet", "Tuned HRNet"]
    const retrieval_methods = ["Approx. kNN", "Euclidean Distance",
                               "Weighted Confidence Score", "Object Keypoint Similarity"]
    const retrieval_databases = ["MS-COCO", "Styled-COCO", "Arch-Data"]

    return(
      <Collapsible trigger={title} className="advanced_settings"
                   onTriggerOpening={this.open}
                   onTriggerClosing={this.close}
                   open={false}>
        <div className="expandable_area">
          <Row className="expandable_row">
            <Col sm={1}></Col>
            <Col sm={4}>
              <CustomDropdown name="Person Detector" id="person_detector"
                              options={person_detectors}
                              update_state={this.props.update_state}/>
            </Col>
            <Col sm={1}></Col>
            <Col sm={4}>
              <CustomDropdown name="Keypont Detector" id="keypoint_detector"
                              options={keypoint_detectors}
                              update_state={this.props.update_state}/>
            </Col>
            <Col sm={1}></Col>
          </Row>
          <Row className="expandable_row">
            <Col sm={1}></Col>
            <Col sm={4}>
              <CustomDropdown name="Retrieval Method" id="retrieval_method"
                              options={retrieval_methods}
                              update_state={this.props.update_state}/>
            </Col>
            <Col sm={1}></Col>
            <Col sm={4}>
              <CustomDropdown name="Retrieval Database" id="retrieval_database"
                              options={retrieval_databases}
                              update_state={this.props.update_state}/>
            </Col>
            <Col sm={1}></Col>
          </Row>
        </div>
      </Collapsible>
    );
  }

}

export default AdvancedSettings
