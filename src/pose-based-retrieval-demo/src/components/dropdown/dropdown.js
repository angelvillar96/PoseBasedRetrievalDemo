import React from "react"
import Form from 'react-bootstrap/Form'

class CustomDropdown extends React.Component{

  constructor(props){
    super(props)

    this.state = {
      cur_state: "",
      id: this.props.id
    }

    this.changed_value = this.changed_value.bind(this)
  }

  async changed_value(e){
    let new_value = e.target.value;
    await this.setState({
      cur_state: new_value
    })

    this.props.update_state(this.state.id, new_value)

  }


  render(){
    return(
      <div>
        <label className="form-label">{this.props.name}:</label>
        <Form.Control as="select" onChange={this.changed_value}>
          {this.props.options.map(option => (
            <option key={option} value={option}>{option}</option>
          ))}
        </Form.Control>
      </div>
  )}

}

export default CustomDropdown


//
