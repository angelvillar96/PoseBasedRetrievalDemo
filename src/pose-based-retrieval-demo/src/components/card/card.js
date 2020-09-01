import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActionArea from '@material-ui/core/CardActionArea';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';


class CustomCard extends React.Component{

  render(){

    return (
      <Card style={{maxWidth: 345}}>
        <CardActionArea>
          <CardMedia
            style={{height: 140}}
            /* title="Hola" */
            image={this.props.img}
          />
          <CardContent>
            <Typography gutterBottom variant="h5" component="h2">
              {this.props.title}
            </Typography>
            <Typography variant="body2" color="textSecondary" component="p">
              {this.props.description}
            </Typography>
          </CardContent>
        </CardActionArea>
        {/*<CardActions>
           <Button size="large" color="primary">
             Try it!
           </Button>
        </CardActions>*/}
      </Card>
    );
  }
}

export default CustomCard;


//
